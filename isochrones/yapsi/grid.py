from isochrones.grid import ModelGrid
from isochrones.isochrone import Isochrone
import os, re, glob
import numpy as np
import pandas as pd
from isochrones.config import ISOCHRONES


class YAPSIModelGrid(ModelGrid):
    
    datadir = os.path.expanduser('~/yapsi')
    default_bands = list('UBVRIJHK')
    default_kwargs = {}
    
    @classmethod
    def get_band(cls, b):
        if b in 'UBVRIJHK':
            return 'w', b
        else:
            raise ValueError('{0} not in YAPSI grids.'.format(b))
            
    def get_filenames(self, phot='w'):
        return glob.glob('{0}/yapsi_{1}_*.dat'.format(self.datadir, phot))

    @classmethod
    def get_feh(cls, filename):
        """
        example filename: yapsi_w_X0p602357_Z0p027643.dat
        """
        m = re.search('X(\dp\d+)_Z(\dp\d+)\.dat', filename)
        if m:
            X = float(m.group(1).replace('p', '.'))
            Z = float(m.group(2).replace('p', '.'))
            Xsun = 0.703812
            Zsun = 0.016188
            return np.log10((Z/X) / (Zsun/Xsun))
        
    @classmethod
    def to_df(cls, filename):

        df = pd.read_table(filename, comment='#',
             names=['age','mass','logTeff','logL','logg','V','UmB','BmV','VmR','VmI','JmK','HmK','VmK'],
             delim_whitespace=True)
        
        df['age'] = np.log10(df['age']*1e9)
        df['feh'] = cls.get_feh(filename)
        df['B'] = df['BmV'] + df['V']
        df['U'] = df['UmB'] + df['B']
        df['R'] = df['V'] - df['VmR']
        df['I'] = df['V'] - df['VmI']
        df['K'] = df['V'] - df['VmK']
        df['H'] = df['K'] - df['HmK']
        df['J'] = df['K'] - df['JmK']
        
        df = df.drop(['UmB','BmV','VmR','VmI','JmK','HmK','VmK'], axis=1)
        return df

    def _get_df(self):
        return self.df_all('w')
    