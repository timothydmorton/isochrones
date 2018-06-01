import os,re, glob
import numpy as np
import pandas as pd
import logging

from ..config import ISOCHRONES
from ..grid import ModelGrid

class DartmouthModelGrid(ModelGrid):
    """Grid of Dartmouth Models.

    The following photometric systems are included::

        phot_systems = ('SDSSugriz','UBVRIJHKsKp','WISE','LSST','UKIDSS')
        phot_bands = dict(SDSSugriz=['sdss_z', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_g'],
              UBVRIJHKsKp=['B', 'I', 'H', 'J', 'Ks', 'R', 'U', 'V', 'D51', 'Kp'],
              WISE=['W4', 'W3', 'W2', 'W1'],
              LSST=['LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'LSST_g', 'LSST_i'],
              UKIDSS=['Y', 'H', 'K', 'J', 'Z'])

    You may add additional ones by putting the ``phot_system.tgz`` file
    into ``.isochrones/dartmouth`` alongside the others.  If you do
    this, you must edit the object definition accordingly so the ``get_band``
    function returns the correct information.
    """
    name = 'dartmouth'
    common_columns = ('EEP', 'MMo', 'LogTeff', 'LogG', 'LogLLo', 'age', 'feh')
    phot_systems = ('SDSSugriz','UBVRIJHKsKp','WISE','LSST','UKIDSS','HST_WFC3')
    phot_bands = dict(SDSSugriz=['sdss_z', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_g'],
                  UBVRIJHKsKp=['B', 'I', 'H', 'J', 'Ks', 'R', 'U', 'V', 'D51', 'Kp'],
                  WISE=['W4', 'W3', 'W2', 'W1'],
                  LSST=['LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'LSST_g', 'LSST_i'],
                  UKIDSS=['Y', 'H', 'K', 'J', 'Z'],
                  HST_WFC3=['uvf200l', 'uvf218w', 'uvf225w', 'uvf275w', 'uvf280n', 'uvf300x',
                            'uvf336w', 'uvf343n', 'uvf350l', 'uvf373n', 'uvf390m', 'uvf390w',
                            'uvf395n', 'uvf410m', 'uvf438w', 'uvf467m', 'uvf469n', 'uvf475w',
                            'uvf475x', 'uvf487n', 'uvf502n', 'uvf547m', 'uvf555w', 'uvf600l',
                            'uvf606w', 'uvf621m', 'uvf625w', 'uvf631n', 'uvf645n', 'uvf656n',
                            'uvf657n', 'uvf658n', 'uvf665n', 'uvf673n', 'uvf680n', 'uvf689m',
                            'uvf763m', 'uvf775w', 'uvf814w', 'uvf845m', 'uvf850l', 'uvf953n',
                            'uvfq232', 'uvfq243', 'uvfq378', 'uvfq387', 'uvfq422', 'uvfq436',
                            'uvfq437', 'uvfq492', 'uvfq508', 'uvfq575', 'uvfq619', 'uvfq634',
                            'uvfq672', 'uvfq674', 'uvfq727', 'uvfq750', 'uvfq889', 'uvfq906',
                            'uvfq924', 'uvfq937', 'irf098m', 'irf105w', 'irf110w', 'irf125w',
                            'irf126n', 'irf127m', 'irf128n', 'irf130n', 'irf132n', 'irf139m',
                            'irf140w', 'irf153m', 'irf160w', 'irf164n', 'irf167n'])

    default_kwargs = {'afe':'afep0', 'y':''}
    datadir = os.path.join(ISOCHRONES, 'dartmouth')
    zenodo_record = 1002927
    zenodo_files = ('dartmouth.tri',)
    zenodo_md5 = ('570b758ea98c8a5a806149bd1b854b98',)
    master_tarball_file = 'dartmouth.tgz'

    default_bands = ('B','V','g','r','i','z',
                     'J','H','K',
                     'W1','W2','W3','Kepler')

    extra_url_base = 'http://stellar.dartmouth.edu/models/isochrones'

    @classmethod
    def get_band(cls, b, **kwargs):
        """Defines what a "shortcut" band name refers to.

        """
        phot = None

        # Default to SDSS for these
        if b in ['u','g','r','i','z']:
            phot = 'SDSSugriz'
            band = 'sdss_{}'.format(b)
        elif b in ['U','B','V','R','I','J','H','Ks']:
            phot = 'UBVRIJHKsKp'
            band = b
        elif b=='K':
            phot = 'UBVRIJHKsKp'
            band = 'Ks'
        elif b in ['kep','Kepler','Kp']:
            phot = 'UBVRIJHKsKp'
            band = 'Kp'
        elif b in ['W1','W2','W3','W4']:
            phot = 'WISE'
            band = b
        elif re.match('uvf', b) or re.match('irf', b):
            phot = 'HST_WFC3'
            band = b
        else:
            m = re.match('([a-zA-Z]+)_([a-zA-Z_]+)',b)
            if m:
                if m.group(1) in cls.phot_systems:
                    phot = m.group(1)
                    if phot=='LSST':
                        band = b
                    else:
                        band = m.group(2)
                elif m.group(1) in ['UK','UKIRT']:
                    phot = 'UKIDSS'
                    band = m.group(2)
        if phot is None:
            raise ValueError('Dartmouth Models cannot resolve band {}!'.format(b))
        return phot, band

    @classmethod
    def phot_tarball_file(cls, phot, **kwargs):
        return os.path.join(cls.datadir, '{}.tgz'.format(phot))

    def get_filenames(self, phot, afe='afep0', y=''):
        if not os.path.exists(os.path.join(self.datadir, 'isochrones', phot)):
            if not os.path.exists(self.phot_tarball_file(phot, afe=afe, y=y)):
                self.extract_phot_tarball(phot)
                # self.extract_master_tarball()
            # self.extract_phot_tarball(phot)

        return glob.glob('{3}/isochrones/{0}/*{1}{2}.{0}*'.format(phot,afe,y,self.datadir))

    @classmethod
    def get_feh(cls, filename):
        m = re.search('feh([mp])(\d+)afe', filename)
        if m:
            sign = 1 if m.group(1)=='p' else -1
            return float(m.group(2))/10 * sign

    @classmethod
    def to_df(cls, filename):
        try:
            rec = np.recfromtxt(filename,skip_header=8,names=True)
        except:
            print('Error reading {}!'.format(filename))
            raise RuntimeError
        df = pd.DataFrame(rec)

        n = len(df)
        ages = np.zeros(n)
        curage = 0
        i=0
        for line in open(filename):
            m = re.match('#',line)
            if m:
                m = re.match('#AGE=\s*(\d+\.\d+)\s+',line)
                if m:
                    curage=float(m.group(1))
            else:
                if re.search('\d',line):
                    ages[i]=curage
                    i+=1

        df['age'] = ages
        df['feh'] = cls.get_feh(filename)
        return df

    def df_all(self, phot):
        df = super(DartmouthModelGrid, self).df_all(phot)
        df.loc[:,'age'] = np.log10(df.age * 1e9) # convert to log10(age)
        df = df.sort_values(by=['feh','age','MMo','EEP'])
        df.index = [df.feh, df.age]
        return df

    def hdf_filename(self, phot):
        afe = self.kwargs['afe']
        y = self.kwargs['y']
        afe_str = '_{}'.format(afe) if afe!='afep0' else ''
        return os.path.join(self.datadir,'{}{}.h5'.format(phot, afe_str, y))
