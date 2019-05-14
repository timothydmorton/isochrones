import re
import logging

import numpy as np
import pandas as pd
try:
    import holoviews as hv
except ImportError:
    hv = None

from . import StarModel, get_ichrone
from .starmodel import SingleStarModel, BinaryStarModel, TripleStarModel
from .priors import PowerLawPrior, FlatLogPrior, FehPrior, FlatPrior, GaussianPrior
from .utils import addmags, band_pairs
from .cluster_utils import calc_lnlike_grid, integrate_over_eeps


def clusterfit(starfile, bands=None, props=None, models='mist', max_distance=10000,
               mineep=200, maxeep=800, maxAV=0.1, minq=0.2,
               overwrite=False, nlive=1000,
               name='', halo_fraction=0.5, comm=None, rank=0, max_iter=0):

    """Fit cluster properties to a table of member stars
    """

    if rank == 0:
        stars = pd.read_hdf(starfile)

        cat = StarCatalog(stars, bands=bands, props=props)
        print('bands = {}'.format(cat.bands))
        print(cat.df.head())

        ic = get_ichrone(models, bands=cat.bands)

        model = StarClusterModel(ic, cat, eep_bounds=(mineep, maxeep),
                                 max_distance=max_distance, minq=minq,
                                 halo_fraction=halo_fraction,
                                 max_AV=maxAV, name=name)

    else:
        model = None

    if comm:
        model = comm.bcast(model, root=0)

    model.fit(overwrite=overwrite, n_live_points=nlive, max_iter=max_iter)


class StarCatalog(object):
    """Catalog of star measurements


    Parameters
    ----------
    df : `pandas.DataFrame`
        Table containing stellar measurements.  Names of uncertainty columns are 
        tagged with `_unc`.  If `bands` is not provided, then names of photometric
        bandpasses will be determined by looking for columns tagged with `_mag`.

    bands ; list(str)
        List of photometric bandpasses in table.  If not provided, will be inferred.

    props : list(str)
        Names of other properties in table (e.g., `Teff`, `logg`, `parallax`, etc.).

    """

    def __init__(self, df, bands=None, props=None):
        self._df = df

        if bands is None:
            bands = []
            for c in df.columns:
                m = re.search('(.+)_mag$', c)
                if m:
                    bands.append(m.group(1))
        self.bands = tuple(bands)
        self.band_cols = tuple('{}_mag'.format(b) for b in self.bands)

        self.props = tuple() if props is None else tuple(props)

        for c in self.band_cols + self.props:
            if c not in self.df.columns:
                raise ValueError('{} not in DataFrame!'.format(c))
            if not '{}_unc'.format(c) in self.df.columns:
                raise ValueError('{0} uncertainty ({0}_unc) not in DataFrame!'.format(c))

        self._ds = None
        self._hr = None

    def __setstate__(self, odict):
        self.__dict__ = odict
        self._hr = None

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, newdf):
        self._df = newdf
        self._ds = None
        self._hr = None

    def get_measurement(self, prop, values=False):
        return self.df[prop].values, self.df[prop + '_unc'].values

    def iter_bands(self, **kwargs):
        for b, col in zip(self.bands, self.band_cols):
            yield b, self.get_measurement(col, **kwargs)

    def iter_props(self, **kwargs):
        for p in self.props:
            yield p, self.get_measurement(p, **kwargs)

    @property
    def ds(self):
        if self._ds is None:
            df = self.df.copy()
            for b1, b2 in band_pairs(self.bands):
                mag1 = self.df['{}_mag'.format(b1)]
                mag2 = self.df['{}_mag'.format(b2)]

                df[b2] = mag2
                df['{0}-{1}'.format(b1, b2)] = mag1 - mag2

            self._ds = hv.Dataset(df)

        return self._ds

    @property
    def hr(self):
        if self._hr is None:
            layout = []
            opts = dict(invert_yaxis=True, tools=['hover'])
            for b1, b2 in band_pairs(self.bands):
                kdims = ['{}-{}'.format(b1, b2), '{}_mag'.format(b1)]
                layout.append(hv.Points(self.ds, kdims=kdims, vdims=self.ds.kdims).options(**opts))
            self._hr = hv.Layout(layout)
        return self._hr

    def iter_models(self, ic=None, N=1):
        if ic is None:
            ic = get_ichrone('mist', bands=self.bands)

        mod_type = {1: SingleStarModel,
                    2: BinaryStarModel,
                    3: TripleStarModel}

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            mags = {b: (row['{}_mag'.format(b)], row['{}_mag_unc'.format(b)]) for b in self.bands}
            props = {p: (row[p], row['{}_unc'.format(p)]) for p in self.props}
            yield mod_type[N](ic, **mags, **props, name=row.name)
            i += 1

    def write_ini(self, ic=None, root='.'):
        if ic is None:
            ic = get_ichrone('mist', bands=self.bands)

        for mod in self.iter_models(ic):
            mod.write_ini(root)

class SimulatedCluster(StarCatalog):
    def __init__(self, N, age, feh, distance, AV, alpha, gamma, fB,
                 bands='JHK', mass_range=(0.3, 2.5), distance_scatter=5,
                 models='mist', phot_unc=0.01):

        self.N = N

        self.age = age
        self.feh = feh
        self.distance = distance
        self.AV = AV
        self.alpha = alpha
        self.gamma = gamma
        self.fB = fB
        self.pars = [age, feh, distance, AV, alpha, gamma, fB]

        self.bands = bands
        self.mass_range = mass_range
        self.distance_scatter = distance_scatter
        self.phot_unc = phot_unc

        self.ic = get_ichrone(models)

        df = self._generate()

        super().__init__(df, bands=bands)

    def evolve(self, age):
        _, feh, distance, AV, alpha, gamma, fB = self.pars

        df = self._simulate_stars(age, self.df.is_binary,
                                  self.df.mass_pri, self.df.mass_sec,
                                  self.df.distance)

        return StarCatalog(df, bands=self.bands)

    def _generate(self):
        N = self.N
        age, feh, distance, AV, alpha, gamma, fB = self.pars

        u = np.random.random(N)
        is_binary = u < fB

        # This is if we want the IMF to describe the primary mass only
        pri_masses = PowerLawPrior(alpha, self.mass_range).sample(N)
        qs = PowerLawPrior(gamma, (0.2, 1)).sample(N)
        sec_masses = pri_masses * qs * is_binary
        sec_masses[(sec_masses < 0.1) & (sec_masses > 0)] = 0.1

        # # This is if we want the IMF to describe the sum of masses
        # tot_masses = PowerLawPrior(alpha, self.mass_range).sample(N)
        # qs = PowerLawPrior(gamma, (0.2, 1)).sample(N)
        # pri_masses = tot_masses / (1 + qs)
        # sec_masses = pri_masses * qs * is_binary
        # sec_masses[(sec_masses < 0.1) & (sec_masses > 0)] = 0.1

        # slightly different distance for each star
        distances = distance + np.random.randn(N) * self.distance_scatter

        return self._simulate_stars(age, is_binary, pri_masses, sec_masses, distances)

    def _simulate_stars(self, age, is_binary, pri_masses, sec_masses, distances):
        N = self.N
        _, feh, distance, AV, alpha, gamma, fB = self.pars

        pri_eeps = np.array([self.ic.get_eep(m, age, feh)
                             for m in pri_masses])
        sec_eeps = np.array([self.ic.get_eep(m, age, feh) if m else 0
                             for m in sec_masses])

        mags = {}
        for b in self.bands:
            pri = np.array([self.ic.mag[b](e, age, feh, d, AV)
                            for e, d in zip(pri_eeps, distances)])
            sec = np.array([self.ic.mag[b](e, age, feh, d, AV) if e else np.inf
                            for e, d in zip(sec_eeps, distances)])
            mags['{}_mag'.format(b)] = addmags(pri, sec)

        stars = pd.DataFrame(mags)

        # Record simulation data
        stars['is_binary'] = is_binary
        stars['distance'] = distances

        stars['mass_pri'] = pri_masses
        stars['mass_sec'] = sec_masses
        stars['eep_pri'] = pri_eeps
        stars['eep_sec'] = sec_eeps

        unc = self.phot_unc
        for b in self.bands:
            stars['{}_mag'.format(b)] += np.random.randn(N) * unc
            stars['{}_mag_unc'.format(b)] = unc

        stars['parallax'] = 1000./distances
        stars['parallax_unc'] = 0.2

        return stars


class StarClusterModel(StarModel):

    param_names = ['age', 'feh', 'distance', 'AV', 'alpha', 'gamma', 'fB']

    def __init__(self, ic, stars, name='',
                 halo_fraction=0.5, max_AV=1., max_distance=50000,
                 use_emcee=False, eep_bounds=None,
                 mass_bounds=None, minq=0.1, **kwargs):
        self._ic = ic

        if not isinstance(stars, StarCatalog):
            stars = StarCatalog(stars, **kwargs)

        self.stars = stars

        self._priors = {'age': FlatLogPrior(bounds=(6, 10.15)),
                        'feh': FehPrior(halo_fraction=halo_fraction),
                        'AV': FlatPrior(bounds=(0, max_AV)),
                        'distance': PowerLawPrior(alpha=2., bounds=(0, max_distance)),
                        'alpha': FlatPrior(bounds=(-4, -1)),
                        'gamma': GaussianPrior(0.3, 0.1),
                        'fB': FlatPrior(bounds=(0., 0.6))}

        self.use_emcee = use_emcee

        self._eep_bounds = eep_bounds
        self._mass_bounds = mass_bounds
        self.minq = minq

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
        if prop == 'eep':
            return self._eep_bounds if self._eep_bounds is not None else (self.ic.mineep,
                                                                          self.ic.maxeep)
        elif prop == 'mass':
            return self._mass_bounds if self._mass_bounds is not None else (self.ic.minmass,
                                                                            self.ic.maxmass)

        try:
            return self._priors[prop].bounds
        except AttributeError:
            if prop == 'age':
                return (self.ic.minage, self.ic.maxage)
            elif prop == 'feh':
                return (self.ic.minfeh, self.ic.maxfeh)
            elif prop == 'gamma':
                return (0, 1)
            elif prop == 'fB':
                return (0, 1)

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
            prior = self._priors[prop]
            if hasattr(prior, 'lnpdf'):
                lnval = prior.lnpdf(eval(prop))
            else:
                lnval = np.log(prior(eval(prop)))

            lnp += lnval

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
        dm_deeps = self.ic.interp_value([eeps, age, feh], ['dm_deep'])
        ln_dm_deeps = np.log(np.absolute(dm_deeps))

        # Compute model mags at each eep
        # model_mags = {b: self.ic.mag[b](eeps, age, feh, distance, AV) for b in self.bands}
        _, _, _, model_mags_arr = self.ic.interp_mag([eeps, age, feh, distance, AV], self.bands)

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
        # model_mags_arr = np.empty((Neep, Nbands), dtype=float)
        vals_arr = np.empty((Nstars, Nbands), dtype=float)
        uncs_arr = np.empty((Nstars, Nbands), dtype=float)
        for i, (b, (vals, uncs)) in enumerate(self.stars.iter_bands(values=True)):
            # model_mags_arr[:, i] = model_mags[b]
            vals_arr[:, i] = vals
            uncs_arr[:, i] = uncs

        args = (lnlike_prop,
                model_mags_arr, Nbands,
                model_masses, ln_dm_deeps, eeps,
                vals_arr, uncs_arr,
                alpha, gamma, fB,
                mass_lo, mass_hi, self.minq)
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

        for i, par in enumerate(['age', 'feh', 'distance', 'AV', 'alpha', 'gamma', 'fB']):
            lo, hi = self.bounds(par)
            cube[i] = (hi - lo)*cube[i] + lo

    def _make_samples(self):

        if not self.use_emcee:
            filename = '{}post_equal_weights.dat'.format(self.mnest_basename)
            try:
                chain = np.loadtxt(filename)
                try:
                    lnprob = chain[:, -1]
                    chain = chain[:, :-1]
                except IndexError:
                    lnprob = np.array([chain[-1]])
                    chain = np.array([chain[:-1]])
            except:
                logging.error('Error loading chains from {}'.format(filename))
                raise
        else:
            chain = self.sampler.flatchain
            lnprob = self.sampler.lnprobability.ravel()

        df = pd.DataFrame(chain, columns=['age', 'feh', 'distance', 'AV', 'alpha', 'gamma', 'fB'])
        df['lnprob'] = lnprob

        self._samples = df


def simulate_cluster(N, age, feh, distance, AV, alpha, gamma, fB,
                     bands='JHK', mass_range=(0.8, 2.5), distance_scatter=5,
                     iso=None):
    u = np.random.random(N)
    is_binary = u < fB

    pri_masses = PowerLawPrior(alpha, mass_range).sample(N)
    qs = PowerLawPrior(gamma, (0.1, 1)).sample(N)
    sec_masses = pri_masses * qs * is_binary

    if iso is None:
        iso = get_ichrone('mist')

    pri_eeps = np.array([iso.get_eep(m, age, feh) for m in pri_masses])
    sec_eeps = np.array([iso.get_eep(m, age, feh, return_nan=True) for m in sec_masses])

    mags = {}
    # slightly different distance for each star
    distances = distance + np.random.randn(N) * distance_scatter
    for b in bands:
        pri = np.array([float(iso.interp_mag([e, age, feh, d, AV], [b])[3]) for e, d in zip(pri_eeps, distances)])
        sec = np.array([float(iso.interp_mag([e, age, feh, d, AV], [b])[3]) for e, d in zip(sec_eeps, distances)])
        sec[~is_binary] = np.inf
        mags['{}_mag'.format(b)] = addmags(pri, sec)

    stars = pd.DataFrame(mags)

    # Record simulation data
    stars['is_binary'] = is_binary
    stars['age'] = age
    stars['feh'] = feh
    stars['distance'] = distances
    stars['AV'] = AV

    stars['mass_pri'] = pri_masses
    stars['mass_sec'] = sec_masses
    stars['eep_pri'] = pri_eeps
    stars['eep_sec'] = sec_eeps

    unc = 0.01
    for b in bands:
        stars['{}_mag'.format(b)] += np.random.randn(N) * unc
        stars['{}_mag_unc'.format(b)] = unc

    stars['distance'] = distances
    stars['parallax'] = 1000./distances
    stars['parallax_unc'] = 0.2

    return StarCatalog(stars, bands=bands, props=['parallax'])
