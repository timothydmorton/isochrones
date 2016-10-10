import os,re
import numpy as np
import pandas as pd
import logging

from .fileutils import get_hdf

def get_grid():
    pass

class ModelGrid(object):
    common_columns = ('EEP', 'MMo', 'LogTeff', 'LogG', 'LogLLo', 'age', 'feh')
    phot_systems = ('SDSSugriz','UBVRIJHKsKp','WISE','LSST','UKIDSS')
    phot_bands = dict(SDSSugriz=['sdss_z', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_g'],
                  UBVRIJHKsKp=['B', 'I', 'H', 'J', 'Ks', 'R', 'U', 'V', 'D51', 'Kp'],
                  WISE=['W4', 'W3', 'W2', 'W1'],
                  LSST=['LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'LSST_g', 'LSST_i'],
                  UKIDSS=['Y', 'H', 'K', 'J', 'Z'])

    default_kwargs = {'afe':'afep0', 'y':''}

    def __init__(self, bands, **kwargs):
        self.bands = bands
        self.kwargs = kwargs

        for k,v in self.default_kwargs.items():
            if k not in self.kwargs:
                self.kwargs[k] = v            

        self._df = None

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

    @property
    def df(self):
        if self._df is None:
            self._df = self._get_df()

        return self._df
    
    def _get_df(self):
        """Returns stellar model grid with desired bandpasses and with standard column names
        
        bands must be iterable, and are parsed according to :func:``get_band``
        """
        grids = {}
        df = pd.DataFrame()
        for bnd in self.bands:
            s,b = self.get_band(bnd)
            logging.debug('loading {} band from {}'.format(b,s))
            if s not in grids:
                grids[s] = get_hdf(s, **self.kwargs)
            if 'MMo' not in df:
                df[list(self.common_columns)] = grids[s][list(self.common_columns)]
            col = grids[s][b]
            n_nan = np.isnan(col).sum()
            if n_nan > 0:
                logging.debug('{} NANs in {} column'.format(n_nan, b))
            df.loc[:, bnd] = col.values #dunno why it has to be this way; something
                                        # funny with indexing.

        return df

