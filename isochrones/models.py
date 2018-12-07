import os
import re
import itertools

import numpy as np
import pandas as pd
from astropy import constants as const

G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value

from .config import ISOCHRONES
from .interp import DFInterpolator
from .mags import interp_mag, interp_mags
from .grid import Grid


class ModelGrid(Grid):

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


class ModelGridInterpolator(object):

    grid_type = None
    bc_type = None

    # transformation from desired param order to that expected by interp functions
    _param_index_order = (1, 2, 0, 3, 4)

    def __init__(self, bands=None):
        self.bands = bands if bands is not None else list(self.bc_type.default_bands)

        self._model_grid = None
        self._bc_grid = None

        self.param_index_order = list(self._param_index_order)

    @property
    def name(self):
        return self.grid_type.name

    @property
    def model_grid(self):
        if self._model_grid is None:
            self._model_grid = self.grid_type()
        return self._model_grid

    @property
    def bc_grid(self):
        if self._bc_grid is None:
            self._bc_grid = self.bc_type(self.bands)
        return self._bc_grid

    def initialize(self, pars=None):
        if pars is None:
            pars = [1.04, 320, -0.35, 10000, 0.34]

        Teff, logg, feh, mags = self.interp_mag([1.04, 320, -0.35, 10000, 0.34], self.bands)
        assert all([np.isfinite(v) for v in [Teff, logg, feh]])
        assert all([np.isfinite(m) for m in mags])

    def _prop(self, prop, *pars):
        return self.interp_value(pars, [prop]).squeeze()

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

    def get_eep(self, mass, age, feh, approx=False):
        return self.model_grid.get_eep(mass, age, feh, approx=approx)

    def model_value(self, mass, age, feh, props):
        if isinstance(props, str):
            props = [props]
        eep = self.get_eep(mass, age, feh)
        values = self.interp_value([mass, eep, feh])
        if np.size(values) == 1:
            return float(values)
        else:
            return values

    def model_mag(self, mass, age, feh, distance=10., AV=0., bands=None):
        if bands is None:
            bands = self.bands
        eep = self.get_eep(mass, age, feh)
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
