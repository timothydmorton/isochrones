from __future__ import print_function, division
import logging 

from .config import on_rtd

if not on_rtd:
    import numpy as np
    from scipy.stats import uniform 
    from scipy.integrate import quad
    import matplotlib.pyplot as plt

class Prior(object):

    def __call__(self, x, **kwargs):
        return self.pdf(x, **kwargs)

    def pdf(self, x):
        raise NotImplementedError

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
        pdf = np.array([quad(self.pdf,lo,hi)[0]/(hi-lo) for lo,hi in zip(b[:-1], b[1:])])
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

    def __call__(self, x):
        lo, hi = self.bounds
        if x < lo or x > hi:
            return 0
        else:
            return self.pdf(x)

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

    def __init__(self, halo_fraction=0.001):   
        self.halo_fraction = halo_fraction
        super(FehPrior, self).__init__()

    def pdf(self, x):
        feh = x

        disk_norm = 2.5066282746310007 # integral of the below from -np.inf to np.inf
        disk_fehdist = 1./disk_norm * (0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
                       +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.))

        halo_mu, halo_sig = -1.5, 0.4
        halo_fehdist = 1./np.sqrt(2*np.pi*halo_sig**2) * np.exp(-0.5*(feh-halo_mu)**2/halo_sig**2)

        return self.halo_fraction * halo_fehdist + (1 - self.halo_fraction)*disk_fehdist

    def sample(self, n):
        w1, mu1, sig1 = 0.8, 0.016, 0.15
        w2, mu2, sig2 = 0.2, -0.15, 0.22

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

#  Uniform true age prior; where 'age' is actually log10(age)
age_prior = FlatLogPrior(bounds=(9, 10.15))
distance_prior = PowerLawPrior(alpha=2., bounds=(0,3000))
AV_prior = FlatPrior(bounds=(0., 1.))
q_prior = PowerLawPrior(alpha=0.3, bounds=(0.1, 1))
salpeter_prior = PowerLawPrior(alpha=-2.35, bounds=(0.1, 10))
feh_prior = FehPrior(halo_fraction=0.001)

