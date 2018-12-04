import os
import re
import glob
import itertools
import logging

import numpy as np
import pandas as pd

from isochrones.config import ISOCHRONES

from ..models import ModelGrid
from .utils import max_eep


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

    @property
    def kwarg_tag(self):
        tag = super().kwarg_tag
        return '{tag}_{kind}'.format(tag=tag, **self.kwargs)

    def get_directory_path(self, **kwargs):
        return os.path.join(self.datadir, 'MIST{}'.format(self.kwarg_tag))

    def get_tarball_file(self, **kwargs):
        filename = self.get_directory_path(**kwargs)
        return '{}.tar.gz'.format(filename)

    def get_tarball_url(self, **kwargs):
        return 'http://waps.cfa.harvard.edu/MIST/data/tarballs' + \
               '_v{version}/MIST_v{version}_vvcrit{vvcrit}_{kind}.tar.gz'.format(**self.kwargs)

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


class MISTEvolutionTrackGrid(MISTModelGrid):
    default_kwargs = {'version': '1.2', 'vvcrit': 0.4, 'afe': 0.0}

    fehs = (-4.00, -3.50, -3.00, -2.50, -2.00,
            -1.75, -1.50, -1.25, -1.00, -0.75, -0.50,
            -0.25, 0.00, 0.25, 0.50)

    index_cols = ('initial_feh', 'initial_mass', 'EEP')

    default_columns = (tuple(set(MISTModelGrid.default_columns) - {'age'}) +
                            ('interpolated', 'star_age', 'age'))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._masses = None

    @property
    def masses(self):
        if self._masses is None:
            self._masses = self.df.index.levels[1]

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
        return 'http://waps.cfa.harvard.edu/MIST/data/tarballs_v{version}/{basename}.tar.gz'.format(version=version, basename=basename)

    def get_tarball_file(self, feh):
        basename = self.get_file_basename(feh)
        return os.path.join(self.datadir, '{}.tar.gz'.format(basename))

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
            for i, m in enumerate(masses):
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
                    new_eeps = np.arange(n_eep, eep_max + 1)
                    new_index = pd.MultiIndex.from_product([[feh], [m], new_eeps])
                    new_data = pd.DataFrame(index=new_index, columns=df_interp.columns, dtype=float)

                    # Interpolate values
                    norm_distance = (m - mlo) / (mhi - mlo)
                    lo_index = pd.MultiIndex.from_product([[feh], [mlo], new_eeps])
                    hi_index = pd.MultiIndex.from_product([[feh], [mhi], new_eeps])
                    new_data.loc[:, df.columns] = df.loc[lo_index, :].values * (1 - norm_distance) + df.loc[hi_index, :].values * norm_distance
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
            for f, m in itertools.product(*df.index.levels[:2]):
                subdf = df.loc[f, m]
                log_age = np.log10(subdf['star_age'])
                deriv = np.gradient(log_age, subdf['eep'])
                subdf.loc[:, 'dt_deep'] = deriv

            df.dt_deep.to_hdf(filename, 'dt_deep')
            dt_deep = pd.read_hdf(filename, 'dt_deep')

        return dt_deep

