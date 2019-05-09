import os
import re
import itertools

import numpy as np
import pandas as pd
from astropy import constants as const
from tqdm import tqdm
from scipy.optimize import minimize

G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value

from .config import ISOCHRONES
from .interp import DFInterpolator, interp_eep, interp_eeps
from .mags import interp_mag, interp_mags
from .utils import addmags
from .grid import Grid


class StellarModelGrid(Grid):

    default_columns = ('eep', 'age', 'feh', 'mass', 'initial_mass', 'radius',
                       'density', 'logTeff', 'Teff', 'logg', 'logL', 'Mbol')


    @property
    def prop_map(self):
        return dict(eep=self.eep_col, age=self.age_col, feh=self.feh_col,
                    mass=self.mass_col, initial_mass=self.initial_mass_col,
                    logTeff=self.logTeff_col, logg=self.logg_col, logL=self.logL_col)

    @property
    def column_map(self):
        return {v: k for k, v in self.prop_map.items()}

    @property
    def datadir(self):
        return os.path.join(ISOCHRONES, self.name)

    @property
    def kwarg_tag(self):
        raise NotImplementedError

    def get_directory_path(self, **kwargs):
        raise NotImplementedError

    def get_existing_filenames(self, **kwargs):
        d = self.get_directory_path(**kwargs)
        if not os.path.exists(d):
            self.extract_tarball(**kwargs)
        return [os.path.join(d, f) for f in os.listdir(d) if re.search(self.filename_pattern, f)]

    def get_filenames(self, **kwargs):
        """ Returns list of all filenames corresponding to phot system and kwargs.
        """
        return self.get_existing_filenames(**kwargs)

    @classmethod
    def get_feh(cls, filename):
        raise NotImplementedError

    @classmethod
    def to_df(cls, filename):
        """Parse raw filename to dataframe
        """
        raise NotImplementedError

    def df_all(self):
        """Entire original model grid as dataframe

        TODO: also save this as HDF, in case it's useful for anything
        """
        df = pd.concat([self.to_df(f) for f in self.get_filenames()])
        df = df.sort_values(by=list(self.index_cols))
        df.index = [df[c] for c in self.index_cols]
        return df

    def compute_additional_columns(self, df):
        """
        """
        df['Teff'] = 10**df['logTeff']
        df['Mbol'] = 4.74 - 2.5 * df['logL']
        df['radius'] = 10**df['log_R']
        df['density'] = df['mass'] * MSUN / (4./3 * np.pi * (df['radius'] * RSUN)**3)
        return df

    def get_df(self, orig=False):
        """Returns column-mapped, pared-down, standardized version of model grid
        """
        df = self.df_all()
        if not orig:
            df = df.rename(columns=self.column_map)
            df = self.compute_additional_columns(df)
            # Select only the columns we want
            df = df[list(self.default_columns)]
        return df

    @property
    def hdf_filename(self):
        return os.path.join(self.datadir, '{}{}.h5'.format(self.name, self.kwarg_tag))

    def get_dm_deep(self, compute=False):
        filename = os.path.join(self.datadir, 'dm_deep{}.h5'.format(self.kwarg_tag))

        compute = not os.path.exists(filename)

        if not compute:
            try:
                dm_deep = pd.read_hdf(filename, 'dm_deep')
            except Exception:
                compute = True

        if compute:
            # need grid to work with first
            df = self.get_df()

            # Make bucket for derivative to go in
            df['dm_deep'] = np.nan

            # Compute derivative for each (feh, age) isochrone, and fill in
            for f, a in itertools.product(*df.index.levels[:2]):
                subdf = df.loc[f, a]
                deriv = np.gradient(subdf['initial_mass'], subdf['eep'])
                subdf.loc[:, 'dm_deep'] = deriv

            df.dm_deep.to_hdf(filename, 'dm_deep')
            dm_deep = pd.read_hdf(filename, 'dm_deep')

        return dm_deep

    @property
    def df(self):
        if self._df is None:
            self._df = self.read_hdf()
            self._df['dm_deep'] = self.get_dm_deep()

        return self._df

    @property
    def interp_grid_npz_filename(self):
        return os.path.join(self.datadir, 'full_grid{}.npz'.format(self.kwarg_tag))

    @property
    def interp_grid_orig_npz_filename(self):
        return os.path.join(self.datadir, 'full_grid_orig{}.npz'.format(self.kwarg_tag))

    def get_array_grids(self, recalc=False):
        calculate = recalc or not os.path.exists(self.array_grid_filename)

        if calculate:
            if self.eep_replaces == 'age':
                ii0 = self.fehs
                ii1 = self.masses
            elif self.eep_replaces == 'mass':
                raise NotImplementedError('Not implemented for isochrone grids yet!')

            n = len(ii0) * len(ii1)
            age_arrays = np.zeros((n, self.n_eep)) * np.nan
            dt_deep_arrays = np.zeros((n, self.n_eep)) * np.nan
            lengths = np.zeros(n) * np.nan
            for i, (x0, x1) in tqdm(enumerate(itertools.product(ii0, ii1)), total=n,
                                    desc='building irregular age grid'):
                subdf = self.df.xs((x0, x1), level=(0, 1))
                xs = subdf[self.eep_replaces].values
                lengths[i] = len(xs)
                try:
                    age_arrays[i, :len(xs)] = xs
                    dt_deep_arrays[i, :len(xs)] = subdf.dt_deep.values
                except ValueError:
                    import pdb
                    pdb.set_trace()

            np.savez(self.array_grid_filename, age=age_arrays, dt_deep=dt_deep_arrays, lengths=lengths.astype(int))

        d = np.load(self.array_grid_filename)

        return d['age'], d['dt_deep'], d['lengths']

    @property
    def array_grid_filename(self):
        return os.path.join(self.datadir, 'array_grid{}.npz'.format(self.kwarg_tag))

    @property
    def age_grid(self):
        try:
            return self._age_grid
        except AttributeError:
            age_grid, dt_deep_grid, lengths = self.get_array_grids()
            self._age_grid = age_grid
            self._dt_deep_grid = dt_deep_grid
            self._array_lengths = lengths
            return self._age_grid

    @property
    def dt_deep_grid(self):
        try:
            return self._dt_deep_grid
        except AttributeError:
            age_grid, dt_deep_grid, lengths = self.get_array_grid()
            self._age_grid = age_grid
            self._dt_deep_grid = arrays
            self._array_lengths = lengths
            return self._dt_deep_grid

    @property
    def array_lengths(self):
        try:
            return self._array_lengths
        except AttributeError:
            age_grid, dt_deep_grid, lengths = self.get_array_grid()
            self._age_grid = age_grid
            self._dt_deep_grid = arrays
            self._array_lengths = lengths
            return self._array_lengths

    @property
    def n_masses(self):
        try:
            return self._n_masses
        except AttributeError:
            self._n_masses = len(self.masses)
            return self._n_masses

class ModelGridInterpolator(object):

    grid_type = None
    bc_type = None

    # transformation from desired param order to that expected by interp functions
    _param_index_order = (1, 2, 0, 3, 4)

    def __init__(self, bands=None, **kwargs):
        self.bands = bands if bands is not None else list(self.bc_type.default_bands)

        self._model_grid = None
        self._bc_grid = None

        self.param_index_order = list(self._param_index_order)

        self.kwargs = kwargs

        self._fehs = None
        self._ages = None
        self._masses = None

    @property
    def minfeh(self):
        return self.model_grid.get_limits('feh')[0]

    @property
    def maxfeh(self):
        return self.model_grid.get_limits('feh')[1]

    @property
    def mineep(self):
        return self.model_grid.get_limits('eep')[0]

    @property
    def maxeep(self):
        return self.model_grid.get_limits('eep')[1]

    @property
    def minage(self):
        return self.model_grid.get_limits('age')[0]

    @property
    def maxage(self):
        return self.model_grid.get_limits('age')[1]

    @property
    def minmass(self):
        return self.model_grid.get_limits('mass')[0]

    @property
    def maxmass(self):
        return self.model_grid.get_limits('mass')[1]

    @property
    def fehs(self):
        if self._fehs is None:
            self._fehs = self.model_grid.fehs
        return self._fehs

    @property
    def ages(self):
        if not self.eep_replaces == 'mass':
            raise AttributeError('Age is not a dimension of model grid type {}!'.format(self.grid_type))
        if self._ages is None:
            self._ages = self.model_grid.ages
        return self._ages

    @property
    def masses(self):
        if not self.eep_replaces == 'age':
            raise AttributeError('Mass is not a dimension of this model grid!'.format(self.grid_type))
        if self._masses is None:
            self._masses = self.model_grid.masses
        return self._masses

    @property
    def name(self):
        return self.grid_type.name

    @property
    def eep_replaces(self):
        return self.grid.eep_replaces

    @property
    def model_grid(self):
        if self._model_grid is None:
            self._model_grid = self.grid_type(**self.kwargs)
        return self._model_grid

    @property
    def bc_grid(self):
        if self._bc_grid is None:
            self._bc_grid = self.bc_type(self.bands)
        return self._bc_grid

    def initialize(self, pars=None):
        if pars is None:
            if self.eep_replaces == 'age':
                pars = [1.04, 320., -0.35, 10000., 0.34]
            elif self.eep_replaces == 'mass':
                pars = [320, 9.7, -0.35, 10000., 0.34]

        Teff, logg, feh, mags = self.interp_mag(pars, self.bands)
        assert all([np.isfinite(v) for v in [Teff, logg, feh]])
        assert all([np.isfinite(m) for m in mags])

    def _prop(self, prop, *pars):
        return self.interp_value(pars, [prop]).squeeze()

    def mass(self, *pars):
        return self._prop('mass', *pars)

    def initial_mass(self, *pars):
        return self._prop('initial_mass', *pars)

    def radius(self, *pars):
        return self._prop('radius', *pars)

    def Teff(self, *pars):
        return self._prop('Teff', *pars)

    def logg(self, *pars):
        return self._prop('logg', *pars)

    def feh(self, *pars):
        return self._prop('feh', *pars)

    def density(self, *pars):
        return self._prop('density', *pars)

    def nu_max(self, *pars):
        return self._prop('nu_max', *pars)

    def delta_nu(self, *pars):
        return self._prop('delta_nu', *pars)

    def interp_value(self, pars, props):
        """

        pars : age, feh, eep, [distance, AV]
        """
        try:
            pars = np.atleast_1d(pars[self.param_index_order])
        except TypeError:
            i0, i1, i2, i3, i4 = self.param_index_order
            pars = [pars[i0], pars[i1], pars[i2]]
        return self.model_grid.interp(pars, props)

    def interp_mag(self, pars, bands):
        """

        pars : age, feh, eep, distance, AV
        """
        if not bands:
            i_bands = np.array([], dtype=int)
        else:
            i_bands = [self.bc_grid.interp.columns.index(b) for b in bands]

        try:
            pars = np.atleast_1d(pars).astype(float).squeeze()
            if pars.ndim > 1:
                raise ValueError
            return interp_mag(pars, self.param_index_order,
                              self.model_grid.interp.grid,
                              self.model_grid.interp.column_index['Teff'],
                              self.model_grid.interp.column_index['logg'],
                              self.model_grid.interp.column_index['feh'],
                              self.model_grid.interp.column_index['Mbol'],
                              *self.model_grid.interp.index_columns,
                              self.bc_grid.interp.grid, i_bands,
                              *self.bc_grid.interp.index_columns)
        except (TypeError, ValueError):
            # Broadcast appropriately.
            b = np.broadcast(*pars)
            pars = np.array([np.resize(x, b.shape).astype(float) for x in pars])
            return interp_mags(pars, self.param_index_order,
                               self.model_grid.interp.grid,
                               self.model_grid.interp.column_index['Teff'],
                               self.model_grid.interp.column_index['logg'],
                               self.model_grid.interp.column_index['feh'],
                               self.model_grid.interp.column_index['Mbol'],
                               *self.model_grid.interp.index_columns,
                               self.bc_grid.interp.grid, i_bands,
                               *self.bc_grid.interp.index_columns)

    def model_value(self, mass, age, feh, props, approx=False):
        if isinstance(props, str):
            props = [props]
        eep = self.get_eep(mass, age, feh, approx=approx)
        values = self.interp_value([mass, eep, feh])
        if np.size(values) == 1:
            return float(values)
        else:
            return values

    def model_mag(self, mass, age, feh, distance=10., AV=0., bands=None, approx=False):
        if bands is None:
            bands = self.bands
        eep = self.get_eep(mass, age, feh, approx=approx)
        pars = [mass, eep, feh, distance, AV]
        _, _, _, mags = self.interp_mag(pars, bands)
        if np.size(mags) == 1:
            return float(mags)
        else:
            return mags

    def __call__(self, p1, p2, p3, distance=10., AV=0.):
        p1, p2, p3, dist, AV = [np.atleast_1d(a).astype(float).squeeze()
                                for a in np.broadcast_arrays(p1, p2, p3, distance, AV)]
        pars = [p1, p2, p3, dist, AV]
        # print(pars)
        prop_cols = self.model_grid.df.columns
        props = self.interp_value(pars, prop_cols)
        _, _, _, mags = self.interp_mag(pars, self.bands)
        cols = list(prop_cols) + ['{}_mag'.format(b) for b in self.bands]
        values = np.concatenate([np.atleast_2d(props), np.atleast_2d(mags)], axis=1)
        return pd.DataFrame(values, columns=cols)

    def isochrone(self, age, feh=0.0, eep_range=None, distance=10., AV=0.0, dropna=True):
        if eep_range is None:
            eep_range = self.model_grid.get_limits('eep')
        eeps = np.arange(*eep_range)

        df = self(eeps, age, feh, distance=distance, AV=AV)
        if dropna:
            return df.dropna()
        else:
            return df

    def mass_age_resid(self, *args, **kwargs):
        raise NotImplementedError

    def max_eep(self, mass, feh):
        return self.model_grid.max_eep(mass, feh)

    def get_eep(self, mass, age, feh, accurate=False, **kwargs):
        grid = self.model_grid
        if ((isinstance(mass, float) or isinstance(mass, int)) and
                (isinstance(age, float) or isinstance(age, int)) and
                (isinstance(feh, float) or isinstance(feh, int))):
            if accurate:
                return self.get_eep_accurate(mass, age, feh, **kwargs)
            else:
                if grid.eep_replaces == 'age':
                    return interp_eep(age, feh, mass, grid.fehs, grid.masses, grid.n_masses,
                                      grid.age_grid, grid.dt_deep_grid, grid.array_lengths)
                elif grid.eep_replaces == 'mass':
                    raise NotImplementedError
        else:
            b = np.broadcast(mass, age, feh)
            pars = [np.atleast_1d(np.resize(x, b.shape)).astype(float)
                    for x in [age, feh, mass]]
            if accurate:
                return np.array([self.get_eep_accurate(m, a, f, **kwargs) for a, f, m in zip(*pars)])
            else:
                if grid.eep_replaces == 'age':
                    return interp_eeps(*pars, grid.fehs, grid.masses, grid.n_masses,
                                       grid.age_grid, grid.dt_deep_grid, grid.array_lengths)
                elif grid.eep_replaces == 'mass':
                    raise NotImplementedError

    def get_eep_accurate(self, mass, age, feh, eep0=300, resid_tol=0.02, method='nelder-mead',
                         return_object=False, return_nan=False, **kwargs):

        eeps_to_try = [min(self.max_eep(mass, feh) - 20, 600), 100, 200]
        while np.isnan(self.mass_age_resid(eep0, mass, age, feh)):
            try:
                eep0 = eeps_to_try.pop()
            except IndexError:
                raise ValueError('eep0 gives nan for all initial guesses! {}'.format((mass, age, feh)))

        result = minimize(self.mass_age_resid, eep0, args=(mass, age, feh), method=method,
                          options=kwargs)

        if return_object:
            return result

        if result.success and result.fun < resid_tol**2:
            return float(result.x)
        else:
            if return_nan:
                return np.nan
            else:
                raise RuntimeError('EEP minimization not successful: {}'.format((mass, age, feh)))

    def generate(self, mass, age, feh, props='all', bands=None,
                 return_df=True, return_dict=False,
                 distance=10, AV=0, accurate=False):
        if bands is None:
            bands = self.bands
        eeps = self.get_eep(mass, age, feh, accurate=accurate)
        values = self.interp_value([mass, eeps, feh], props)
        if bands:
            _, _, _, mags = self.interp_mag([mass, eeps, feh, distance, AV], bands=bands)
            axis = 1 if values.ndim == 2 else 0
            values = np.concatenate([values, mags], axis=axis)

        if return_dict:
            if props == 'all':
                props = self.model_grid.interp.columns + ['{}_mag'.format(b) for b in bands]
            values = dict(zip(props, values))
        elif return_df:
            if props == 'all':
                props = self.model_grid.interp.columns
            values = pd.DataFrame(np.atleast_2d(values),
                                  columns=props + ['{}_mag'.format(b) for b in bands])

        return values


    def generate_binary(self, mass_A, mass_B, age, feh, **kwargs):
        bands = kwargs.get('bands', None)
        if bands is None:
            bands = self.bands

        values_A = self.generate(mass_A, age, feh, **kwargs)
        values_B = self.generate(mass_B, age, feh, **kwargs)

        if isinstance(values_A, pd.DataFrame):
            column_map_A = {c: '{}_0'.format(c) for c in values_A.columns}
            column_map_B = {c: '{}_1'.format(c) for c in values_B.columns}

            values = pd.concat([values_A.rename(columns=column_map_A),
                                values_B.rename(columns=column_map_B)], axis=1)

            for b in bands:
                values['{}_mag'.format(b)] = addmags(values_A['{}_mag'.format(b)],
                                                     values_B['{}_mag'.format(b)])

        return values


class EvolutionTrackInterpolator(ModelGridInterpolator):
    param_names = ('mass', 'eep', 'feh', 'distance', 'AV')
    eep_replaces = 'age'

    # Relation between parameters and the order of indices in the grid
    _param_index_order = (2, 0, 1, 3, 4)
    _iso_type = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iso = None

    @property
    def iso(self):
        if self._iso is None:
            if self._iso_type is None:
                raise ValueError('{} has no _iso_type!.'.format(type(self)))
            self._iso = self._iso_type(bands=self.bands)
        return self._iso

    def mass_age_resid(self, eep, mass, age, feh):
        # mass_interp = self.iso.interp_value([eep, age, feh], ['mass'])
        age_interp = self.interp_value([mass, eep, feh], ['age'])
        # return (mass - mass_interp)**2 + (age - age_interp)**2
        return (age - age_interp)**2


class IsochroneInterpolator(ModelGridInterpolator):
    param_names = ('eep', 'age', 'feh', 'distance', 'AV')
    eep_replaces = 'mass'

    # Relation between parameters and the order of indices in the grid
    _param_index_order = (1, 2, 0, 3, 4)
    _track_type = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None

    @property
    def track(self):
        if self._track is None:
            if self._track_type is None:
                raise ValueError('{} has no _track_type!'.format(type(self)))
            self._track = self._track_type(bands=self.bands)
        return self._track

    def mass_age_resid(self, eep, mass, age, feh):
        mass_interp = self.interp_value([eep, age, feh], ['initial_mass'])
        # age_interp = self.track.interp_value([mass, eep, feh], ['age'])
        # return (mass - mass_interp)**2 + (age - age_interp)**2
        return (mass - mass_interp)**2

    def generate(self, *args, **kwargs):
        return self.track.generate(*args, **kwargs)
