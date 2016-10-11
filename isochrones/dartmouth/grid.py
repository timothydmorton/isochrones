import os,re, glob
import numpy as np
import pandas as pd
import logging

from ..config import ISOCHRONES
from ..grid import ModelGrid

class DartmouthModelGrid(ModelGrid):
    name = 'dartmouth'
    common_columns = ('EEP', 'MMo', 'LogTeff', 'LogG', 'LogLLo', 'age', 'feh')
    phot_systems = ('SDSSugriz','UBVRIJHKsKp','WISE','LSST','UKIDSS')
    phot_bands = dict(SDSSugriz=['sdss_z', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_g'],
                  UBVRIJHKsKp=['B', 'I', 'H', 'J', 'Ks', 'R', 'U', 'V', 'D51', 'Kp'],
                  WISE=['W4', 'W3', 'W2', 'W1'],
                  LSST=['LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'LSST_g', 'LSST_i'],
                  UKIDSS=['Y', 'H', 'K', 'J', 'Z'])

    default_kwargs = {'afe':'afep0', 'y':''}
    datadir = os.path.join(ISOCHRONES, 'dartmouth')
    zenodo_record = 159426
    zenodo_files = ('dartmouth.tgz', 'dartmouth.tri')
    master_tarball_file = 'dartmouth.tgz'

    def get_band(self, b):
        """Defines what a "shortcut" band name refers to.  Returns phot_system, band

        """
        # Default to SDSS for these
        if b in ['u','g','r','i','z']:
            sys = 'SDSSugriz'
            band = 'sdss_{}'.format(b)
        elif b in ['U','B','V','R','I','J','H','Ks']:
            sys = 'UBVRIJHKsKp'
            band = b
        elif b=='K':
            sys = 'UBVRIJHKsKp'
            band = 'Ks'
        elif b in ['kep','Kepler','Kp']:
            sys = 'UBVRIJHKsKp'
            band = 'Kp'
        elif b in ['W1','W2','W3','W4']:
            sys = 'WISE'
            band = b
        else:
            m = re.match('([a-zA-Z]+)_([a-zA-Z_]+)',b)
            if m:
                if m.group(1) in self.phot_systems:
                    sys = m.group(1)
                    if sys=='LSST':
                        band = b
                    else:
                        band = m.group(2)
                elif m.group(1) in ['UK','UKIRT']:
                    sys = 'UKIDSS'
                    band = m.group(2)
        return sys, band

    def phot_tarball_file(self, phot, **kwargs):
        return os.path.join(self.datadir, '{}.tgz'.format(phot))

    def get_filenames(self, phot, afe='afep0', y=''):
        if not os.path.exists(os.path.join(self.datadir, 'isochrones', phot)):
            if not os.path.exists(self.phot_tarball_file(phot, afe=afe, y=y)):
                self.extract_master_tarball()
            self.extract_phot_tarball(phot)

        return glob.glob('{3}/isochrones/{0}/*{1}{2}.{0}*'.format(phot,afe,y,self.datadir))

    def get_feh(self, filename):
        m = re.search('feh([mp])(\d+)afe', filename)
        if m:
            sign = 1 if m.group(1)=='p' else -1
            return float(m.group(2))/10 * sign
        
    def to_df(self, filename):
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
        df['feh'] = self.get_feh(filename)
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

