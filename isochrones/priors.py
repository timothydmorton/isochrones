from __future__ import print_function, division

from .config import on_rtd

if not on_rtd:
    import numpy as np


class Prior(object):

    def __call__(self, x, **kwargs):
        return self._func(x, **kwargs)

    def _func(self, x):
        raise NotImplementedError

class BoundedPrior(Prior):
    def __init__(self, bounds=None):
        self.bounds = bounds

    def __call__(self, x, bounds=None, **kwargs):
        lo, hi = self.bounds if bounds is None else bounds
        if x < lo or x > hi:
            return 0
        else:
            return self._func(x, bounds, **kwargs)


class FlatPrior(BoundedPrior):

    def _func(self, x, bounds):
        lo, hi = bounds
        return 1./(hi - lo)

class FlatLogPrior(BoundedPrior):

    def _func(self, x, bounds):
        lo, hi = bounds
        return np.log(10) * 10**x / (10**hi - 10**lo)

class PowerLawPrior(BoundedPrior):
    def __init__(self, alpha, bounds=None):
        self.alpha = alpha
        super(PowerLawPrior, self).__init__(bounds=bounds)

    def _func(self, x, bounds):
        lo, hi = bounds
        C = (1 + self.alpha)/(hi**(1 + self.alpha) - lo**(1 + self.alpha))
        # C = 1/(1/(self.alpha+1)*(1 - lo**(self.alpha+1)))
        return C * x**self.alpha

class DistancePrior(BoundedPrior):
    """Uniform spherical distribution
    """
    def _func(self, x, bounds):
        _, hi = bounds
        return  3/hi**3 * x**2

class FehPrior(Prior):
    """feh PDF based on local SDSS distribution
    
    From Jo Bovy:
    https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
    2D gaussian fit based on Casagrande (2011)
    """

    def __init__(self, halo_fraction=0.001):   
        self.halo_fraction = halo_fraction

    def _func(self, x, **kwargs):
        feh = x

        disk_norm = 2.5066282746310007 # integral of the below from -np.inf to np.inf
        disk_fehdist = 1./disk_norm * (0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
                       +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.))

        halo_mu, halo_sig = -1.5, 0.4
        halo_fehdist = 1./np.sqrt(2*np.pi*halo_sig**2) * np.exp(-0.5*(feh-halo_mu)**2/halo_sig**2)

        return self.halo_fraction * halo_fehdist + (1 - self.halo_fraction)*disk_fehdist


#  Uniform true age prior; where 'age' is actually log10(age)
age_prior = FlatLogPrior(bounds=(9, 10.15))
distance_prior = DistancePrior(bounds=(0,3000))
AV_prior = FlatPrior(bounds=(0., 1.))
q_prior = PowerLawPrior(alpha=0.3, bounds=(0.1, 1))
salpeter_prior = PowerLawPrior(alpha=-2.35, bounds=(0.1, 10))
feh_prior = FehPrior(halo_fraction=0.001)

