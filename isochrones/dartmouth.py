from __future__ import division,print_function
import os,os.path
import numpy as np
import pkg_resources

import pandas as pd
import pickle

from .isochrones import Isochrone

#DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
DATADIR = os.path.expanduser('~/.isochrones')
if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)

MASTERFILE = '{}/dartmouth.h5'.format(DATADIR)
TRI_FILE = '{}/dartmouth.tri'.format(DATADIR)

def _download_h5():
    url = 'www.figshare.com/s/095e235e70ca11e4b89506ec4b8d1f61'
    import urllib2
    print('Downloading Dartmouth stellar model data (should happen only once)...')
    u = urllib2.urlopen(url)
    f = open(MASTERFILE,'wb')
    f.write(u.read())
    f.close()

def _download_tri():
    url = 'www.figshare.com/s/f840a5f670c911e49c8c06ec4b8d1f61'
    import urllib2
    print('Downloading Dartmouth isochrone pre-computed triangulation (should happen only once...')
    u = urllib2.urlopen(url)
    f = open(TRI_FILE,'wb')
    f.write(u.read())
    f.close()

if not os.path.exists(MASTERFILE):
    _download_h5()

if not os.path.exists(TRI_FILE):
    _download_tri()

MASTERDF = pd.read_hdf(MASTERFILE,'df')

class Dartmouth_Isochrone(Isochrone):
    """Dotter (2008) Stellar Models
    """
    def __init__(self,bands=['U','B','V','R','I','J','H','K','g','r','i','z','Kepler','D51']):

        df = MASTERDF

        mags = {}
        for band in bands:
            try:
                mags[band] = df[band]
            except:
                if band == 'kep' or band == 'Kepler':
                    mags[band] = df['Kp']
                elif band == 'K':
                    mags['K'] = df['Ks']
                else:
                    raise

        f = open(TRI_FILE,'rb')
        tri = pickle.load(f)
        f.close()
        
        Isochrone.__init__(self,df['M'],np.log10(df['age']*1e9),
                           df['feh'],df['M'],df['logL'],
                           10**df['logTeff'],df['logg'],mags,tri=tri)


