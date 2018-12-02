from __future__ import print_function, division
import logging

from .config import on_rtd

if not on_rtd:
    import numpy as np
    import pandas as pd
    import scipy.stats
    from scipy.stats import uniform
    from scipy.integrate import quad
    from scipy.stats._continuous_distns import _norm_pdf, _norm_cdf, _norm_logpdf

    import matplotlib.pyplot as plt
    from numba import jit
    from math import log, log10


_norm_pdf_C = np.sqrt(2*np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)

def _norm_pdf(x):
    return np.exp(-x**2/2.0) / _norm_pdf_C


def _norm_logpdf(x):
    return -x**2 / 2.0 - _norm_pdf_logC


class Prior(object):

    def __call__(self, x, **kwargs):
        return self.pdf(x, **kwargs)

    def pdf(self, x):
        raise NotImplementedError

    def lnpdf(self, x, **kwargs):
        if hasattr(self, '_lnpdf'):
            return self._lnpdf(x, **kwargs)
        else:
            pdf = self(x, **kwargs)
            return np.log(pdf) if pdf else -np.inf

    def sample(self, n):
        raise NotImplementedError

    def test_integral(self):
        assert np.isclose(1, quad(self.pdf, -np.inf, np.inf)[0])

    def test_sampling(self, n=100000, plot=False):
        x = self.sample(n)
        if hasattr(self, 'bounds'):
            rng = self.bounds
        else:
            rng = None
        hn, _ = np.histogram(x, range=rng)
        h, b = np.histogram(x, density=True, range=rng)
        logging.debug('bins: {}'.format(b))
        logging.debug('raw: {}'.format(hn))
        pdf = np.array([quad(self.pdf, lo, hi)[0]/(hi-lo) for lo, hi in zip(b[:-1], b[1:])])
        sigma = 1./np.sqrt(hn)
        resid = np.absolute(pdf - h) / pdf
        logging.debug('pdf: {}'.format(pdf))
        logging.debug('hist: {}'.format(h))
        logging.debug('sigma: {}'.format(sigma))
        logging.debug('{}'.format(resid/sigma))

        c = (b[:-1] + b[1:])/2
        if plot:
            plt.plot(c, h, '_')
            plt.plot(c, pdf, 'x')
        else:
            assert max((resid / sigma)[hn > 50]) < 4

class BoundedPrior(Prior):
    def __init__(self, bounds=None):
        self.bounds = bounds
        super(BoundedPrior, self).__init__()

    def test_integral(self):
        assert np.isclose(1, quad(self.pdf, *self.bounds)[0])

    def __call__(self, x, **kwargs):
        if self.bounds is not None:
            lo, hi = self.bounds
            if x < lo or x > hi:
                return 0
        return self.pdf(x, **kwargs)

    def lnpdf(self, x, **kwargs):
        if self.bounds is not None:
            lo, hi = self.bounds
            if x < lo or x > hi:
                return -np.inf
        if hasattr(self, '_lnpdf'):
            return self._lnpdf(x, **kwargs)
        else:
            pdf = self.pdf(x, **kwargs)
            return np.log(pdf) if pdf else -np.inf


class GaussianPrior(BoundedPrior):
    def __init__(self, mean, sigma, bounds=None):
        self.mean = mean
        self.sigma = sigma
        self.bounds = bounds

        if bounds:
            lo, hi = bounds
            a, b = (lo - mean) / sigma, (hi - mean) / sigma
            self.distribution = scipy.stats.truncnorm(a, b, loc=mean, scale=sigma)
            self.norm = _norm_cdf(b) - _norm_cdf(a)
            self.lognorm = np.log(self.norm)
        else:
            self.distribution = scipy.stats.norm(mean, sigma)
            self.norm = 1.
            self.lognorm = 0.

    def pdf(self, x):
        return _norm_pdf((x - self.mean) / self.sigma) / self.sigma / self.norm

    def _lnpdf(self, x):
        return _norm_logpdf((x - self.mean) / self.sigma) - np.log(self.sigma) - self.lognorm

    def sample(self, n):
        return self.distribution.rvs(n)


class FlatPrior(BoundedPrior):

    def __init__(self, bounds):
        super(FlatPrior, self).__init__(bounds=bounds)

    def pdf(self, x):
        lo, hi = self.bounds
        return 1./(hi - lo)

    def sample(self, n):
        lo, hi = self.bounds
        return np.random.random(n)*(hi - lo) + lo

class FlatLogPrior(BoundedPrior):
    def __init__(self, bounds):
        super(FlatLogPrior, self).__init__(bounds=bounds)

    def pdf(self, x):
        lo, hi = self.bounds
        return np.log(10) * 10**x/ (10**hi - 10**lo)

    def sample(self, n):
        lo, hi = self.bounds
        return np.log10(np.random.random(n)*(10**hi - 10**lo) + 10**lo)

class PowerLawPrior(BoundedPrior):
    def __init__(self, alpha, bounds=None):
        self.alpha = alpha
        super(PowerLawPrior, self).__init__(bounds=bounds)

    def pdf(self, x):
        lo, hi = self.bounds
        C = (1 + self.alpha)/(hi**(1 + self.alpha) - lo**(1 + self.alpha))
        # C = 1/(1/(self.alpha+1)*(1 - lo**(self.alpha+1)))
        return C * x**self.alpha

    def _lnpdf(self, x):
        lo, hi = self.bounds
        C = (1 + self.alpha)/(hi**(1 + self.alpha) - lo**(1 + self.alpha))
        return np.log(C) + self.alpha*np.log(x)

    def sample(self, n):
        """

        cdf = C * int(x**a)|x=lo..x

            = C * [x**(a+1) / (a+1)] | x=lo..x
            = C * ([x**(a+1) / (a+1)] - [lo**(a+1) / (a+1)])
         u  =

         u/C + [lo**(a+1) / (a+1)] = x**(a+1) / (a+1)
         (a+1) * (u/C + [lo**(a+1) / (a+1)]) = x**(a+1)
         [(a+1) * (u/C + [lo**(a+1) / (a+1)])] ** (1/(a+1)) = x
        """
        lo, hi = self.bounds
        C = (1 + self.alpha)/(hi**(1 + self.alpha) - lo**(1 + self.alpha))
        u = np.random.random(n)
        a = self.alpha
        return ((a+1) * (u/C + (lo**(a+1) / (a+1))))**(1/(a+1))


class FehPrior(Prior):
    """feh PDF based on local SDSS distribution

    From Jo Bovy:
    https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
    2D gaussian fit based on Casagrande (2011)
    """

    def __init__(self, halo_fraction=0.001, local=True):
        self.halo_fraction = halo_fraction
        self.local = local

        super(FehPrior, self).__init__()

    def pdf(self, x):
        feh = x

        if self.local:
            disk_norm = 2.5066282746310007  # integral of the below from -np.inf to np.inf
            disk_fehdist = 1./disk_norm * (0.8/0.15*np.exp(-0.5*(feh - 0.016)**2./0.15**2.) +
                                           0.2/0.22*np.exp(-0.5*(feh + 0.15)**2./0.22**2.))
        else:
            mu, sig = -0.3, 0.3
            disk_fehdist = 1./np.sqrt(2*np.pi)/sig * np.exp(-0.5 * (feh - mu)**2 / sig**2)

        halo_mu, halo_sig = -1.5, 0.4
        halo_fehdist = 1./np.sqrt(2*np.pi*halo_sig**2) * np.exp(-0.5*(feh-halo_mu)**2/halo_sig**2)

        return self.halo_fraction * halo_fehdist + (1 - self.halo_fraction)*disk_fehdist

    def sample(self, n):
        if self.local:
            w1, mu1, sig1 = 0.8, 0.016, 0.15
            w2, mu2, sig2 = 0.2, -0.15, 0.22
        else:
            w1, mu1, sig1 = 1.0, -0.3, 0.3
            w2, mu2, sig2 = 0.0, 0, 1

        halo_mu, halo_sig = -1.5, 0.4

        x1 = np.random.randn(n)*sig1 + mu1
        x2 = np.random.randn(n)*sig2 + mu2

        xhalo = np.random.randn(n)*halo_sig + halo_mu

        u1 = np.random.random(n)
        x = x1
        m1 = u1 < w2
        x[m1] = x2[m1]

        u2 = np.random.random(n)
        m2 = u2 < self.halo_fraction
        x[m2] = xhalo[m2]
        return x


class EEP_prior(BoundedPrior):
    def __init__(self, ic, orig_prior, bounds=None):
        self.ic = ic
        self.orig_prior = orig_prior
        self.bounds = bounds if bounds is not None else ic.eep_bounds
        self.orig_par = ic.eep_replaces
        if self.orig_par == 'age':
            self.deriv_prop = 'dt_deep'
        elif self.orig_par == 'mass':
            self.deriv_prop = 'dm_deep'
        else:
            raise ValueError('wtf.')

    def pdf(self, eep, **kwargs):
        if self.orig_par == 'age':
            pars = [kwargs['mass'], eep, kwargs['feh']]
        elif self.orig_par == 'mass':
            pars = [eep, kwargs['age'], kwargs['feh']]
        orig_val, dx_deep = self.ic.interp_value(pars, [self.orig_par, self.deriv_prop]).squeeze()
        return self.orig_prior(orig_val) * dx_deep

    def sample(self, n, **kwargs):
        eeps = pd.Series(np.arange(self.bounds[0], self.bounds[1])).sample(n, replace=True)

        if self.orig_par == 'age':
            mass = kwargs['mass']
            if isinstance(mass, pd.Series):
                mass = mass.values
            feh = kwargs['feh']
            if isinstance(feh, pd.Series):
                feh = feh.values
            values = self.ic.interp_value([mass, eeps, feh], ['dt_deep', 'age'])
            deriv_val, orig_val = values[:, 0], values[:, 1]
            weights = np.log10(orig_val)/np.log10(np.e) * deriv_val
        elif self.orig_par == 'mass':
            age = kwargs['age']
            if isinstance(age, pd.Series):
                age = age.values
            feh = kwargs['feh']
            if isinstance(feh, pd.Series):
                feh = feh.values
            values = self.ic.interp_value([eeps, age, feh], ['dm_deep', 'mass'])
            deriv_val, orig_val = values[:, 0], values[:, 1]
            raise NotImplementedError('Implement proper weighting for EEP-replace-mass sampling!')

        return eeps.sample(n, weights=weights, replace=True).values


# Utility numba PDFs for speed!
@jit(nopython=True)
def powerlaw_pdf(x, alpha, lo, hi):
    alpha_plus_one = alpha + 1
    C = alpha_plus_one/(hi**alpha_plus_one - lo**alpha_plus_one)
    return C * x**alpha

@jit(nopython=True)
def powerlaw_lnpdf(x, alpha, lo, hi):
    alpha_plus_one = alpha + 1
    C = alpha_plus_one/(hi**alpha_plus_one - lo**alpha_plus_one)
    return log(C) + alpha * log(x)



#  Uniform true age prior; where 'age' is actually log10(age)
age_prior = FlatLogPrior(bounds=(5, 10.15))
distance_prior = PowerLawPrior(alpha=2., bounds=(0,10000))
AV_prior = FlatPrior(bounds=(0., 1.))
q_prior = PowerLawPrior(alpha=0.3, bounds=(0.1, 1))
salpeter_prior = PowerLawPrior(alpha=-2.35, bounds=(0.1, 10))
feh_prior = FehPrior(halo_fraction=0.001)
