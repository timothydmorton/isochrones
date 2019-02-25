import os
import re
import glob
import itertools
import logging
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit

from isochrones.config import ISOCHRONES

from ..models import ModelGrid
from ..eep import fit_section_poly, eep_fn, eep_jac, eep_fn_p0
from .eep import max_eep
from ..interp import DFInterpolator, searchsorted
from ..utils import polyval
from .eep import max_eep


class MISTModelGrid(ModelGrid):
    name = 'mist'
    eep_col = 'EEP'
    age_col = 'log10_isochrone_age_yr'
    feh_col = '[Fe/H]'
    mass_col = 'star_mass'
    initial_mass_col = 'initial_mass'
    logTeff_col = 'log_Teff'
    logg_col = 'log_g'
    logL_col = 'log_L'

    default_kwargs = {'version': '1.2', 'vvcrit': 0.0, 'kind': 'full_isos'}
    default_columns = ModelGrid.default_columns + ('delta_nu', 'nu_max', 'phase')

    bounds = (('age', (5, 10.13)),
              ('feh', (-4, 0.5)),
              ('eep', (0, 1710)),
              ('mass', (0.1, 300)))

    fehs = np.array((-4.00, -3.50, -3.00, -2.50, -2.00,
            -1.75, -1.50, -1.25, -1.00, -0.75, -0.50,
            -0.25, 0.00, 0.25, 0.50))
    n_fehs = 15

    primary_eeps = (1, 202, 353, 454, 605, 631, 707, 808, 1409, 1710)
    n_eep = 1710

    def max_eep(self, mass, feh):
        return max_eep(mass, feh)

    @property
    def eep_sections(self):
        return [(a, b) for a, b in zip(self.primary_eeps[:-1], self.primary_eeps[1:])]

    @property
    def kwarg_tag(self):
        return '_v{version}_vvcrit{vvcrit}'.format(**self.kwargs)

    def compute_additional_columns(self, df):
        """
        """
        df = super().compute_additional_columns(df)
        df['feh'] = df['log_surf_z'] - np.log10(df['surface_h1']) - np.log10(0.0181)  # Aaron Dotter says
        return df


class MISTIsochroneGrid(MISTModelGrid):
    eep_col = 'EEP'
    age_col = 'log10_isochrone_age_yr'
    feh_col = '[Fe/H]'
    mass_col = 'star_mass'
    initial_mass_col = 'initial_mass'
    logTeff_col = 'log_Teff'
    logg_col = 'log_g'
    logL_col = 'log_L'

    default_kwargs = {'version': '1.2', 'vvcrit': 0.4, 'kind': 'full_isos'}
    index_cols = ('log10_isochrone_age_yr', 'feh', 'EEP')

    filename_pattern = '\.iso'
    eep_replaces = 'mass'

    @property
    def kwarg_tag(self):
        tag = super().kwarg_tag
        return '{tag}_{kind}'.format(tag=tag, **self.kwargs)

    def get_directory_path(self, **kwargs):
        return os.path.join(self.datadir, 'MIST{}'.format(self.kwarg_tag))

    def get_tarball_file(self, **kwargs):
        filename = self.get_directory_path(**kwargs)
        return '{}.txz'.format(filename)

    def get_tarball_url(self, **kwargs):
        """
        e.g.
        http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_full_isos.txz
        """
        return 'http://waps.cfa.harvard.edu/MIST/data/tarballs' + \
               '_v{version}/MIST_v{version}_vvcrit{vvcrit}_{kind}.txz'.format(**self.kwargs)

    @classmethod
    def get_feh(cls, filename):
        m = re.search('feh_([mp])([0-9]\.[0-9]{2})_afe', filename)
        if m:
            sign = 1 if m.group(1) == 'p' else -1
            return float(m.group(2)) * sign
        else:
            raise ValueError('{} not a valid MIST file? Cannnot parse [Fe/H]'.format(filename))

    @classmethod
    def to_df(cls, filename):
        with open(filename, 'r', encoding='latin-1') as fin:
            while True:
                line = fin.readline()
                if re.match('# EEP', line):
                    column_names = line[1:].split()
                    break
        feh = cls.get_feh(filename)
        df = pd.read_table(filename, comment='#', delim_whitespace=True,
                           skip_blank_lines=True, names=column_names)
        df['feh'] = feh
        return df


class MISTBasicIsochroneGrid(MISTIsochroneGrid):

    default_kwargs = {'version': '1.2', 'vvcrit': 0.4, 'kind': 'basic_isos'}
    default_columns = ModelGrid.default_columns + ('phase',)

    def compute_additional_columns(self, df):
        """
        """
        df = ModelGrid.compute_additional_columns(self, df)
        # df['feh'] = df['log_surf_z'] - np.log10(df['surface_h1']) - np.log10(0.0181)  # Aaron Dotter says
        return df


class MISTEvolutionTrackGrid(MISTModelGrid):
    default_kwargs = {'version': '1.2', 'vvcrit': 0.4, 'afe': 0.0}


    index_cols = ('initial_feh', 'initial_mass', 'EEP')

    default_columns = (tuple(set(MISTModelGrid.default_columns) - {'age'}) +
                            ('interpolated', 'star_age', 'age'))

    eep_replaces = 'age'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fehs = None
        self._masses = None

        self._approx_eep_interp = None
        self._eep_interps = None
        self._primary_eeps_arr = None

    @property
    def masses(self):
        if self._masses is None:
            self._masses = np.array(self.df.index.levels[1])
        return self._masses

    # @property
    # def fehs(self):
    #     if self._fehs is None:
    #         self._fehs = np.array(self.df.index.levels[0])
    #     return self._fehs

    @property
    def datadir(self):
        return os.path.join(ISOCHRONES, self.name, 'tracks')

    @property
    def kwarg_tag(self):
        return '_v{version}_vvcrit{vvcrit}'.format(**self.kwargs)

    @property
    def prop_map(self):
        return dict(eep=self.eep_col,
                    mass=self.mass_col, initial_mass=self.initial_mass_col,
                    logTeff=self.logTeff_col, logg=self.logg_col, logL=self.logL_col)

    def compute_additional_columns(self, df):
        """
        """
        df = super().compute_additional_columns(df)
        df['age'] = np.log10(df['star_age'])
        return df

    def get_file_basename(self, feh):
        feh_sign = 'm' if feh < 0 else 'p'
        afe = self.kwargs['afe']
        afe_sign = 'm' if afe < 0 else 'p'
        fmt_dict = self.kwargs.copy()
        fmt_dict.update(dict(feh=abs(feh), feh_sign=feh_sign,
                             afe_sign=afe_sign, afe=abs(self.kwargs['afe'])))
        return 'MIST_v{version}_feh_{feh_sign}{feh:.2f}_afe_{afe_sign}{afe:.1f}_vvcrit{vvcrit:.1f}_EEPS'.format(**fmt_dict)

    def get_directory_path(self, feh):
        basename = self.get_file_basename(feh)
        return os.path.join(self.datadir, basename)

    def get_tarball_url(self, feh):
        basename = self.get_file_basename(feh)
        version = self.kwargs['version']
        return 'http://waps.cfa.harvard.edu/MIST/data/tarballs_v{version}/{basename}.txz'.format(version=version, basename=basename)

    def get_tarball_file(self, feh):
        basename = self.get_file_basename(feh)
        return os.path.join(self.datadir, '{}.txz'.format(basename))

    def download_and_extract_all(self):
        for feh in self.fehs:
            self.extract_tarball(feh=feh)

    @classmethod
    def get_mass(cls, filename):
        m = re.search('(\d{5})M.track.eep', filename)
        if m:
            return float(m.group(1))/100.
        else:
            raise ValueError('Cannot parse mass from {}.'.format(filename))

    @classmethod
    def to_df(cls, filename):
        with open(filename, 'r', encoding='latin-1') as fin:
            while True:
                line = fin.readline()
                if re.match('^# EEPs', line):
                    line = line.split()
                    eep_first = int(line[2])
                    eep_last = int(line[-1])
                elif re.match('#\s+ star_age', line):
                    column_names = line[1:].split()
                    break
        initial_mass = cls.get_mass(filename)
        df = pd.read_table(filename, comment='#', delim_whitespace=True,
                           skip_blank_lines=True, names=column_names)
        df['initial_mass'] = initial_mass
        try:
            df['EEP'] = np.arange(eep_first, eep_last+1, dtype=int)
        except ValueError:
            print('len(df) is {}; first, last eeps are {}, {} ({})'.format(len(df), eep_first,
                                                                           eep_last, filename))
        return df

    def get_feh_filenames(self, feh):
        directory = self.get_directory_path(feh)
        if not os.path.exists(directory):
            self.extract_tarball(feh=feh)
        return glob.glob(os.path.join(directory, '*.track.eep'))

    def get_feh_hdf_filename(self, feh):
        directory = self.get_directory_path(feh)
        return os.path.join(directory, 'all_masses.h5')

    def get_feh_interpolated_hdf_filename(self, feh):
        directory = self.get_directory_path(feh)
        return os.path.join(directory, 'all_masses_interpolated.h5')

    def df_all_feh(self, feh):
        hdf_filename = self.get_feh_hdf_filename(feh)
        if os.path.exists(hdf_filename):
            df = pd.read_hdf(hdf_filename, 'df')
        else:
            df = pd.concat([self.to_df(f) for f in self.get_feh_filenames(feh)])
            df['initial_feh'] = feh
            df = df.sort_values(by=list(self.index_cols))
            df.index = [df[c] for c in self.index_cols]
            df.to_hdf(hdf_filename, 'df')
            df = pd.read_hdf(hdf_filename, 'df')
        return df

    def df_all_feh_interpolated(self, feh):
        """Same as df_all_feh but with missing track tails interpolated
        """
        hdf_filename = self.get_feh_interpolated_hdf_filename(feh)
        if os.path.exists(hdf_filename):
            df_interp = pd.read_hdf(hdf_filename, 'df')
        else:
            logging.info('Interpolating incomplete tracks for feh = {}'.format(feh))
            df = self.df_all_feh(feh)
            df_interp = df.copy()
            df_interp['interpolated'] = False
            masses = df.index.levels[1]
            for i, m in tqdm(enumerate(masses), total=len(masses),
                             desc="interpolating missing values in evolution tracks (feh={})'".format(feh)):
                n_eep = len(df.xs(m, level='initial_mass'))
                eep_max = max_eep(m, feh)
                if not eep_max:
                    raise ValueError('No eep_max return value for ({}, {})?'.format(m, feh))
                if n_eep < eep_max:

                    # Find lower limit
                    ilo = i
                    found_lower = False
                    while not found_lower:
                        ilo -= 1
                        mlo = masses[ilo]
                        nlo = len(df.xs(mlo, level='initial_mass'))
                        if nlo >= eep_max:
                            found_lower = True
                        if ilo == 0:
                            raise ValueError('Did not find mlo for ({}, {})'.format(m, feh))

                    # Find upper limit
                    ihi = i
                    found_upper = False
                    while not found_upper:
                        ihi += 1
                        mhi = masses[ihi]
                        nhi = len(df.xs(mhi, level='initial_mass'))
                        if nhi >= eep_max:
                            found_upper = True
                        if ihi > len(masses):
                            raise ValueError('Did not find mhi for ({}, {})'.format(m, feh))

                    logging.info('{}: {} (expected {}).  Interpolating between {} and {}'.format(m, n_eep, eep_max, mlo, mhi))
                    new_eeps = np.arange(n_eep + 1, eep_max + 1)
                    new_index = pd.MultiIndex.from_product([[feh], [m], new_eeps])
                    new_data = pd.DataFrame(index=new_index, columns=df_interp.columns, dtype=float)

                    # Interpolate values
                    norm_distance = (m - mlo) / (mhi - mlo)
                    lo_index = pd.MultiIndex.from_product([[feh], [mlo], new_eeps])
                    hi_index = pd.MultiIndex.from_product([[feh], [mhi], new_eeps])
                    new_data.loc[:, df.columns] = (df.loc[lo_index, :].values * (1 - norm_distance) +
                                                   df.loc[hi_index, :].values * norm_distance)
                    new_data.loc[:, 'interpolated'] = True
                    df_interp = pd.concat([df_interp, new_data])

            df_interp.sort_index(inplace=True)
            df_interp.to_hdf(hdf_filename, 'df')
            df_interp = pd.read_hdf(hdf_filename, 'df')

        return df_interp

    def df_all(self):
        df = pd.concat([self.df_all_feh_interpolated(feh) for feh in self.fehs])
        return df

    @property
    def df(self):
        if self._df is None:
            self._df = self.read_hdf()
            self._df['dt_deep'] = self.get_dt_deep()

        return self._df

    def get_dt_deep(self, compute=False):
        filename = os.path.join(self.datadir, 'dt_deep{}.h5'.format(self.kwarg_tag))

        compute = not os.path.exists(filename)

        if not compute:
            try:
                dt_deep = pd.read_hdf(filename, 'dt_deep')
            except Exception:
                compute = True

        if compute:
            # need grid to work with first
            df = self.get_df()

            # Make bucket for derivative to go in
            df['dt_deep'] = np.nan

            # Compute derivative for each (feh, age) isochrone, and fill in
            for f, m in tqdm(itertools.product(*df.index.levels[:2]),
                             total=len(list(itertools.product(*df.index.levels[:2]))),
                             desc='Computing dt/deep'):
                subdf = df.loc[f, m]
                log_age = np.log10(subdf['star_age'])
                deriv = np.gradient(log_age, subdf['eep'])
                subdf.loc[:, 'dt_deep'] = deriv

            df.dt_deep.to_hdf(filename, 'dt_deep')
            dt_deep = pd.read_hdf(filename, 'dt_deep')

        return dt_deep

    @property
    def eep_param_filename(self):
        return os.path.join(self.datadir, 'eep_params{}.h5'.format(self.kwarg_tag))

    def fit_eep_section(self, a, b, order=3):
        fehs = self.df.index.levels[0]
        ms = self.df.index.levels[1]
        columns = ['p{}'.format(o) for o in range(order + 1)]
        p_df = pd.DataFrame(index=pd.MultiIndex.from_product((fehs, ms)), columns=columns)

        for feh, m in tqdm(itertools.product(fehs, ms),
                           total=len(fehs)*len(ms),
                           desc='Fitting age-eep relation for eeps {:.0f} to {:.0f} (order {})'.format(a, b, order)):
            subdf = self.df.xs((feh, m), level=('initial_feh', 'initial_mass'))
            try:
                p = fit_section_poly(subdf.age.values, subdf.eep.values, a, b, order)
            except (TypeError, ValueError):
                p = [np.nan] * (order + 1)
            for c, n in zip(p, range(order + 1)):
                p_df.at[(feh, m), 'p{}'.format(n)] = c
        return p_df

    def fit_approx_eep(self, max_fit_eep=808):
        fehs = self.df.index.levels[0]
        ms = self.df.index.levels[1]
        columns = ['p5', 'p4', 'p3', 'p2', 'p1', 'p0', 'A', 'x0', 'tau']
        par_df = pd.DataFrame(index=pd.MultiIndex.from_product((fehs, ms)), columns=columns)
        for feh, m in tqdm(itertools.product(fehs, ms),
                           total=len(fehs)*len(ms),
                           desc='Fitting approximate eep(age) function'):
            subdf = self.df.xs((feh, m), level=('initial_feh', 'initial_mass'))
            p0 = eep_fn_p0(subdf.age, subdf.eep)
            last_pfit = p0
            mask = subdf.eep < max_fit_eep
            try:
                if subdf.eep.max() < 500:
                    raise RuntimeError
                pfit, _ = curve_fit(eep_fn, subdf.age.values[mask], subdf.eep.values[mask], p0, jac=eep_jac)
            except RuntimeError:  # if the full fit barfs, just use the polynomial by setting A to zero, and the rest same as previous.
                pfit = list(np.polyfit(subdf.age.values[mask], subdf.eep.values[mask], 5)) + last_pfit[-3:]
                pfit[-3] = 0
            last_pfit = pfit
            par_df.loc[(feh, m), :] = pfit
        return par_df.astype(float)

    def write_eep_params(self, orders=None):
        if orders is None:
            orders = [7]*2 + [3] + [1]*6

        p_dfs = [self.fit_eep_section(a, b, order=o) for (a, b), o in zip(self.eep_sections, orders)]
        for df, (a, b) in zip(p_dfs, self.eep_sections):
            df.to_hdf(self.eep_param_filename, 'eep_{:.0f}_{:.0f}'.format(a, b))

        p_approx_df = self.fit_approx_eep()
        p_approx_df.to_hdf(self.eep_param_filename, 'approx')

    def get_eep_interps(self):
        """Get list of interp functions for piecewise polynomial params
        """
        if not os.path.exists(self.eep_param_filename):
            self.write_eep_params()

        with pd.HDFStore(self.eep_param_filename) as store:
            interps = [DFInterpolator(store['eep_{:.0f}_{:.0f}'.format(a, b)]) for a, b in self.eep_sections]
        return interps

    def get_approx_eep_interp(self):
        if not os.path.exists(self.eep_param_filename):
            self.write_eep_params()

        with pd.HDFStore(self.eep_param_filename) as store:
            interp = DFInterpolator(store['approx'])

        return interp

    @property
    def approx_eep_interp(self):
        if self._approx_eep_interp is None:
            self._approx_eep_interp = self.get_approx_eep_interp()

        return self._approx_eep_interp

    @property
    def eep_interps(self):
        if self._eep_interps is None:
            self._eep_interps = self.get_eep_interps()

        return self._eep_interps

    @property
    def primary_eeps_arr(self):
        if self._primary_eeps_arr is None:
            self._primary_eeps_arr = np.array(self.primary_eeps)
        return self._primary_eeps_arr

    def get_eep_fit(self, mass, age, feh, approx=False):
        eep_fn_pars = self.approx_eep_interp([feh, mass], 'all')
        eep = eep_fn(age, *eep_fn_pars)
        if approx:
            return eep
        else:
            i, _ = searchsorted(self.primary_eeps_arr, eep)
            try:
                return polyval(self.eep_interps[i-1]([feh, mass], 'all'), age)
            except IndexError:
                if age > eep_fn_pars[-2]:
                    return polyval(self.eep_interps[-1]([feh, mass], 'all'), age)  # assume you're in last bit
                else:
                    logging.warning('EEP conversion failed for mass={}, age={}, feh={} (approx eep = {}).  Returning nan.'.format(mass, age, feh, eep))
                    return np.nan

    def view_eep_fit(self, mass, feh, plot_fit=True, order=5, p0=None, plot_p0=False):
        import holoviews as hv
        hv.extension('bokeh')
        subdf = self.df.xs((mass, feh), level=('initial_mass', 'initial_feh'))

        ds = hv.Dataset(subdf)
        pts = hv.Points(ds, kdims=['age', 'eep'], vdims=['phase', 'interpolated']).options(tools=['hover'], width=800, height=400, marker='+')
        primary_eeps = self.primary_eeps
        primary_ages = [subdf.loc[e].age for e in primary_eeps if e < subdf.eep.max()]

        from isochrones.eep import eep_fn, eep_jac, eep_fn_p0
        from scipy.optimize import curve_fit
        if p0 is None:
            p0 = eep_fn_p0(subdf.age.values, subdf.eep.values, order=order)

        m = subdf.eep < 808
        if plot_fit:
            pfit, _ = curve_fit(partial(eep_fn, order=order), subdf.age.values[m], subdf.eep.values[m], p0, jac=partial(eep_jac, order=order))
            fit = hv.Points([(a, eep_fn(a, *pfit)) for a in subdf.age])
        if plot_p0:
            p0_fit = hv.Points([(a, eep_fn(a, *p0)) for a in subdf.age])

        olay = pts * hv.Points([(a, e) for a, e in zip(primary_ages, primary_eeps)]).options(size=8)
        if plot_fit:
            olay = olay * fit
        if plot_p0:
            olay = olay * p0_fit
        return olay
