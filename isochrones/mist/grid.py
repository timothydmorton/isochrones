import os,re, glob
import numpy as np
import pandas as pd
import logging
import tarfile
from distutils.version import StrictVersion

from ..config import ISOCHRONES
from ..grid import ModelGrid

class MISTModelGrid(ModelGrid):
    name = 'mist'
    common_columns = ('EEP', 'log10_isochrone_age_yr', 'initial_mass',
                        'log_Teff', 'log_g', 'log_L', 'Z_surf', 'feh', 'phase')

    phot_systems = ('CFHT', 'DECam', 'GALEX', 'JWST', 'LSST', 'PanSTARRS',
                    'SDSS', 'SPITZER', 'SkyMapper', 'UBVRIplus', 'UKIDSS', 'WISE')

    phot_bands = dict(UBVRIplus=['Bessell_U', 'Bessell_B', 'Bessell_V',
                        'Bessell_R', 'Bessell_I', '2MASS_J', '2MASS_H', '2MASS_Ks',
                        'Kepler_Kp', 'Kepler_D51', 'Hipparcos_Hp',
                        'Tycho_B', 'Tycho_V', 'Gaia_G'],
                      WISE=['WISE_W1', 'WISE_W2', 'WISE_W3', 'WISE_W4'],
                      CFHT=['CFHT_u', 'CFHT_g', 'CFHT_r',
                            'CFHT_i_new', 'CFHT_i_old', 'CFHT_z'],
                      DECam=['DECam_u', 'DECam_g', 'DECam_r',
                             'DECam_i', 'DECam_z', 'DECam_Y'],
                      GALEX=['GALEX_FUV', 'GALEX_NUV'],
                      JWST=['F070W', 'F090W', 'F115W', 'F140M',
                           'F150W2', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W',
                           'F210M', 'F212N', 'F250M', 'F277W', 'F300M', 'F322W2', 'F323N',
                           'F335M', 'F356W', 'F360M', 'F405N', 'F410M', 'F430M', 'F444W',
                           'F460M', 'F466N', 'F470N', 'F480M'],
                      LSST=['LSST_u', 'LSST_g', 'LSST_r',
                            'LSST_i', 'LSST_z', 'LSST_y'],
                      PanSTARRS=['PS_g', 'PS_r', 'PS_i', 'PS_z',
                                 'PS_y', 'PS_w', 'PS_open'],
                      SkyMapper=['SkyMapper_u', 'SkyMapper_v', 'SkyMapper_g',
                                 'SkyMapper_r', 'SkyMapper_i', 'SkyMapper_z'],
                      SPITZER=['IRAC_3.6', 'IRAC_4.5', 'IRAC_5.8', 'IRAC_8.0'],
                      UKIDSS=['UKIDSS_Z', 'UKIDSS_Y', 'UKIDSS_J',
                                'UKIDSS_H', 'UKIDSS_K'],
                      SDSS=['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'])

    default_kwargs = {'version':'1.1', 'vvcrit':0.0}
    datadir = os.path.join(ISOCHRONES, 'mist')
    zenodo_record = 161241
    zenodo_files = ()#('mist.tgz',)
    zenodo_md5 = ('0deaaca2836c7148c27ce5ba5bbdfe59',)
    master_tarball_file = 'mist.tgz'

    default_bands = ('G','B','V','J','H','K','W1','W2','W3','g','r','i','z','Kepler')

    def __init__(self, *args, **kwargs):
        version = kwargs.get('version', self.default_kwargs['version'])
        version = StrictVersion(str(version))

        if version > '1.0':
            self.default_bands = self.default_bands + ('TESS', 'BP', 'RP')

        super().__init__(*args, **kwargs)

    @classmethod
    def get_common_columns(cls, version=None, **kwargs):
        if version is None:
            version = cls.default_kwargs['version']

        version = StrictVersion(str(version))

        if version == '1.0':
            return ('EEP', 'log10_isochrone_age_yr', 'initial_mass',
                        'log_Teff', 'log_g', 'log_L', 'Z_surf', 'feh', 'phase')
        elif version >= '1.1':
            return ('EEP', 'log10_isochrone_age_yr', 'initial_mass', 'star_mass',
                    'log_Teff', 'log_g', 'log_L', '[Fe/H]_init', '[Fe/H]')

    @property
    def version(self):
        return StrictVersion(str(self.kwargs['version']))

    @property
    def common_columns(self):
        return self.get_common_columns(self.version)

    def phot_tarball_url(self, phot):
        version = self.kwargs['version']
        vvcrit = self.kwargs['vvcrit']
        url = 'http://waps.cfa.harvard.edu/MIST/data/tarballs_v{0}/MIST_v{0}_vvcrit{1}_{2}.tar.gz'.format(version, vvcrit, phot)
        return url

    @classmethod
    def get_band(cls, b, **kwargs):
        """Defines what a "shortcut" band name refers to.  Returns phot_system, band

        """
        phot = None

        # Default to SDSS for these
        if b in ['u','g','r','i','z']:
            phot = 'SDSS'
            band = 'SDSS_{}'.format(b)
        elif b in ['U','B','V','R','I']:
            phot = 'UBVRIplus'
            band = 'Bessell_{}'.format(b)
        elif b in  ['J','H','Ks']:
            phot = 'UBVRIplus'
            band = '2MASS_{}'.format(b)
        elif b=='K':
            phot = 'UBVRIplus'
            band = '2MASS_Ks'
        elif b in ['kep','Kepler','Kp']:
            phot = 'UBVRIplus'
            band = 'Kepler_Kp'
        elif b=='TESS':
            phot = 'UBVRIplus'
            band = 'TESS'
        elif b in ['W1','W2','W3','W4']:
            phot = 'WISE'
            band = 'WISE_{}'.format(b)
        elif b in ('G', 'BP', 'RP'):
            phot = 'UBVRIplus'
            band = 'Gaia_{}'.format(b)
            if 'version' in kwargs:
                if kwargs['version']=='1.1':
                    band += '_DR2Rev'
        else:
            m = re.match('([a-zA-Z]+)_([a-zA-Z_]+)',b)
            if m:
                if m.group(1) in cls.phot_systems:
                    phot = m.group(1)
                    if phot=='PanSTARRS':
                        band = 'PS_{}'.format(m.group(2))
                    else:
                        band = m.group(0)
                elif m.group(1) in ['UK','UKIRT']:
                    phot = 'UKIDSS'
                    band = 'UKIDSS_{}'.format(m.group(2))

        if phot is None:
            for system, bands in cls.phot_bands.items():
                if b in bands:
                    phot = system
                    band = b
                    break
            if phot is None:
                raise ValueError('MIST grids cannot resolve band {}!'.format(b))
        return phot, band

    def phot_tarball_file(self, phot):
        version = self.kwargs['version']
        vvcrit = self.kwargs['vvcrit']
        return os.path.join(self.datadir, 'MIST_v{}_vvcrit{}_{}.tar.gz'.format(version, vvcrit, phot))

    # def extract_phot_tarball(self, phot):
    #     version = self.kwargs['version']
    #     phot_tarball = self.phot_tarball_file(phot)
    #     with tarfile.open(phot_tarball) as tar:
    #         logging.info('Extracting {}...'.format(phot_tarball))
    #         tar.extractall(self.datadir)

    def get_filenames(self, phot):
        version = self.kwargs['version']
        vvcrit = self.kwargs['vvcrit']
        d = os.path.join(self.datadir, 'MIST_v{}_vvcrit{}_{}'.format(version, vvcrit, phot))
        if not os.path.exists(d):
            if not os.path.exists(self.phot_tarball_file(phot)):
                self.extract_phot_tarball(phot)

        return [os.path.join(d,f) for f in os.listdir(d) if re.search('\.cmd$', f)]

    @classmethod
    def get_feh(cls, filename):
        m = re.search('feh_([mp])([0-9]\.[0-9]{2})_afe', filename)
        if m:
            sign = 1 if m.group(1)=='p' else -1
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

    def df_all(self, phot, **kwargs):
        df = super(MISTModelGrid, self).df_all(phot)
        df = df.sort_values(by=['feh','log10_isochrone_age_yr','initial_mass'])
        df.index = [df.feh, df.log10_isochrone_age_yr]
        return df

    def hdf_filename(self, phot):
        version = self.kwargs['version']
        vvcrit = self.kwargs['vvcrit']
        return os.path.join(self.datadir, 'MIST_v{}_vvcrit{}_{}.h5'.format(version, vvcrit, phot))
