import re

import numpy as np
import pandas as pd

from . import StarModel, get_ichrone
from .priors import PowerLawPrior, FlatLogPrior, FehPrior, FlatPrior
from .utils import addmags
from .cluster_utils import calc_lnlike_grid, integrate_over_eeps

class StarCatalog(object):
    """
    """
    def __init__(self, df, bands=None, props=None):
        self.df = df

        self.bands = tuple() if bands is None else tuple(bands)
        self.props = tuple() if props is None else tuple(props)

        for c in self.bands + self.props:
            if not c in self.df.columns:
                raise ValueError('{} not in DataFrame!'.format(c))
            if not '{}_unc'.format(c) in self.df.columns:
                raise ValueError('{0} uncertainty ({0}_unc) not in DataFrame!'.format(c))

    def get_measurement(self, prop, values=False):
        return self.df[prop].values, self.df[prop + '_unc'].values

    def iter_bands(self, **kwargs):
        for b in self.bands:
            yield b, self.get_measurement(b, **kwargs)

    def iter_props(self, **kwargs):
        for p in self.props:
            yield p, self.get_measurement(p, **kwargs)


class StarClusterModel(StarModel):

    param_names = ['age', 'feh', 'distance', 'AV', 'alpha', 'gamma', 'fB']

    def __init__(self, ic, stars, name='',
                 halo_fraction=0.001, max_AV=1., max_distance=50000,
                 use_emcee=False, eep_bounds=None,
                 mass_bounds=None, qlo=0.1, **kwargs):
        self._ic = ic

        if not isinstance(stars, StarCatalog):
            stars = StarCatalog(stars, **kwargs)

        self.stars = stars

        self._priors = {'age': FlatLogPrior((6, 10.15)),
                       'feh': FehPrior(halo_fraction=halo_fraction),
                       'AV' : FlatPrior((0, max_AV)),
                       'distance' : PowerLawPrior(alpha=2., bounds=(0, max_distance)),
                       'alpha' : FlatPrior((-4, -1)),
                       'gamma' : FlatPrior((0, 1)),
                       'fB' : FlatPrior((0, 1))}

        self.use_emcee = use_emcee

        self._eep_bounds = eep_bounds
        self._mass_bounds = mass_bounds
        self.qlo = qlo

        self.name = name

        self._samples = None
        self._directory = '.'

    @property
    def bands(self):
        return self.stars.bands

    @property
    def props(self):
        return self.stars.props


    @property
    def labelstring(self):
        s = 'cluster'
        if self.name:
            s += '_{}'.format(self.name)
        return s


    def bounds(self, prop):
        if prop=='eep':
            return self._eep_bounds if self._eep_bounds is not None else (self.ic.mineep,
                                                                          self.ic.maxeep)
        elif prop=='mass':
            return self._mass_bounds if self._mass_bounds is not None else (self.ic.minmass,
                                                                            self.ic.maxmass)

        try:
            return self._priors[prop].bounds
        except AttributeError:
            if prop=='age':
                return (self.ic.minage, self.ic.maxage)
            elif prop=='feh':
                return (self.ic.minfeh, self.ic.maxfeh)

    @property
    def n_params(self):
        return len(self.param_names)

    def lnprior(self, p):
        age = p[0]
        feh = p[1]
        distance = p[2]
        AV = p[3]
        alpha = p[4]
        gamma = p[5]
        fB = p[6]

        lnp = 0
        for prop in ['age', 'feh', 'distance', 'AV', 'alpha', 'gamma', 'fB']:
            val = np.log(self._priors[prop](eval(prop)))
            lnp += val

        if not np.isfinite(lnp):
            return -np.inf

        return lnp

    def lnlike(self, p):
        age = p[0]
        feh = p[1]
        distance = p[2]
        AV = p[3]
        alpha = p[4]
        gamma = p[5]
        fB = p[6]

        mineep, maxeep = self.bounds('eep')
        eeps = np.arange(mineep, maxeep + 1)

        # Compute log-likelihood of each mass under power-law distribution
        #  Also use this opportunity to find the valid range of EEP
        model_masses = self.ic.initial_mass(eeps, age, feh)
        ok = np.isfinite(model_masses)

        model_masses = model_masses[ok]
        eeps = eeps[ok]

        # Compute model mags at each eep
        model_mags = {b : self.ic.mag[b](eeps, age, feh, distance, AV) for b in self.bands}

        # Compute the log-likelihood of the (non-mag) props
        #  This needs to be added (appropriately) to lnlike_photmass
        #  As computed here, lnlike_prop is shape Neep x Nstars

        Nstars = len(self.stars.df)
        lnlike_prop = np.zeros((len(eeps), Nstars))
        for p, (vals, uncs) in self.stars.iter_props(values=True):
            if p == 'parallax':
                plax = 1000./distance
                model_vals = np.ones(len(eeps)) * plax # kludge-y
            else:
                model_fn = getattr(self.ic, p)
                model_vals = model_fn(eeps, age, feh)

            lnlike_prop +=  -0.5 * (vals - model_vals[:, None])**2 / uncs**2

        # Create lnlike grid.  This will be a Nstars x Neep x Neep array.

        # Lots of different shaped input arrays here:
        # lnlike_prop: (Nstars, Neep)
        # model_mags: (Neep, Nbands)
        # Nbands: int
        # masses: Neep
        # eeps: Neep
        # mag_values: (Nstars, Nbands)
        # mag_uncs: (Nstars, Nbands)

        # This takes the lnlike_prop and adds likelihoods from photometry and mass
        mass_lo, mass_hi = self.bounds('mass')
        Neep = len(eeps)
        Nbands = len(self.stars.bands)
        model_mags_arr = np.empty((Neep, Nbands), dtype=float)
        vals_arr = np.empty((Nstars, Nbands), dtype=float)
        uncs_arr = np.empty((Nstars, Nbands), dtype=float)
        for i, (b, (vals, uncs)) in enumerate(self.stars.iter_bands(values=True)):
            model_mags_arr[:, i] = model_mags[b]
            vals_arr[:, i] = vals
            uncs_arr[:, i] = uncs

        args = (lnlike_prop,
                model_mags_arr, Nbands,
                model_masses, eeps,
                vals_arr, uncs_arr,
                alpha, gamma, fB,
                mass_lo, mass_hi, self.qlo)
        lnlike_grid = calc_lnlike_grid(*args)


        # Integrate over eep1, eep2
        like_tot = integrate_over_eeps(lnlike_grid, eeps, Nstars)

        if (like_tot == 0).any():
            return -np.inf
        lnlike = np.log(like_tot).sum()
        return lnlike

    def emcee_p0(self, n_walkers):
        raise NotImplementedError('Must provide p0 to fit_mcmc for now.')

    def mnest_prior(self, cube, ndim, nparams):

        for i, par in enumerate(['age','feh','distance','AV', 'gamma']):
            lo, hi = self.bounds(par)
            cube[i] = (hi - lo)*cube[i] + lo

    def _make_samples(self):

        if not self.use_emcee:
            filename = '{}post_equal_weights.dat'.format(self.mnest_basename)
            try:
                chain = np.loadtxt(filename)
                try:
                    lnprob = chain[:,-1]
                    chain = chain[:,:-1]
                except IndexError:
                    lnprob = np.array([chain[-1]])
                    chain = np.array([chain[:-1]])
            except:
                logging.error('Error loading chains from {}'.format(filename))
                raise
        else:
            chain = self.sampler.flatchain
            lnprob = self.sampler.lnprobability.ravel()

        df = pd.DataFrame(chain, columns=['age', 'feh', 'distance', 'AV', 'gamma'])
        df['lnprob'] = lnprob

        self._samples = df

def simulate_cluster(N, age, feh, distance, AV, alpha, gamma, fB, bands='JHK'):
    u = np.random.random(N)
    is_binary = u < fB

    pri_masses = PowerLawPrior(-3, (0.5, 2)).sample(N)
    qs = PowerLawPrior(gamma, (0.1, 1)).sample(N)
    sec_masses = pri_masses * qs * is_binary

    mist = get_ichrone('mist')

    pri_eeps = np.array([mist.eep_from_mass(m, age, feh) for m in pri_masses])
    sec_eeps = np.array([mist.eep_from_mass(m, age, feh) for m in sec_masses])

    mags = {}
    for b in bands:
        pri = np.array([mist.mag[b](e, age, feh, distance, AV) for e in pri_eeps])
        sec = np.array([mist.mag[b](e, age, feh, distance, AV) for e in sec_eeps])
        sec[~is_binary] = np.inf
        mags[b] = addmags(pri, sec)

    stars = pd.DataFrame(mags)
    stars['is_binary'] = is_binary

    unc = 0.02
    for b in bands:
        stars[b] += np.random.randn(N)*0.01
        stars[b + '_unc'] = 0.01

    # slightly different distance for each star
    distances = distance + np.random.randn(N) * 5
    stars['parallax'] = 1000./distances
    stars['parallax_unc'] = 0.2

    return StarCatalog(stars, bands=bands, props=['parallax'])
