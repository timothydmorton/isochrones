import os,re
import pandas as pd

# 
d = os.getenv('ISOCHRONES',
                    os.path.expanduser(os.path.join('~','.isochrones')))
DATADIR = os.path.join(d,'dartmouth')

# Columns in DataFrames that are *not* magnitudes
COMMON_COLUMNS = ['EEP', 'MMo', 'LogTeff', 'LogG', 'LogLLo', 'age', 'feh']

# Default available photometric systems
PHOT_SYSTEMS = ['SDSSugriz','UBVRIJHKsKp','WISE','LSST','UKIDSS']

# Names of bands in each Dartmouth photometric system grid
PHOT_BANDS = dict(SDSSugriz=['sdss_z', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_g'],
                  UBVRIJHKsKp=['B', 'I', 'H', 'J', 'Ks', 'R', 'U', 'V', 'D51', 'Kp'],
                  WISE=['W4', 'W3', 'W2', 'W1'],
                  LSST=['LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'LSST_g', 'LSST_i'],
                  UKIDSS=['Y', 'H', 'K', 'J', 'Z'])

def get_band(b):
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
        m = re.match('(\w+)_([a-zA-Z_]+)',b)
        if m:
            if m.group(1) in PHOT_SYSTEMS:
                sys = m.group(1)
                band = m.group(2)
            elif m.group(1) in ['UK','UKIRT']:
                sys = 'UKIDSS'
                band = m.group(2)
    return sys, band

def get_grid(bands):
    """Returns stellar model grid with desired bandpasses and with standard column names
    
    bands must be iterable, and are parsed according to :func:``get_band``
    """
    grids = {}
    df = pd.DataFrame()
    for bnd in bands:
        s,b = get_band(bnd)
        if s not in grids:
            grids[s] = pd.read_hdf(os.path.join(DATADIR, '{}.h5'.format(s)))
        if 'MMo' not in df:
            df[COMMON_COLUMNS] = grids[s][COMMON_COLUMNS]       
        df[bnd] = grids[s][b]

    return df

