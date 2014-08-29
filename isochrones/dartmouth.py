from __future__ import division,print_function
import os,os.path
import numpy as np
import pkg_resources

import pandas as pd
import pickle

from . import isochrones as iso

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

MASTERFILE = '{}/dartmouth.h5'.format(DATADIR)
MASTERDF = pd.read_hdf(MASTERFILE,'all')
TRI_FILE = '{}/dartmouth.tri'.format(DATADIR)

class Dartmouth_Isochrone(iso.Isochrone):
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
        

        iso.Isochrone.__init__(self,df['M'],np.log10(df['age']*1e9),df['feh'],df['M'],df['logL'],
                               10**df['logTeff'],df['logg'],mags,tri=tri)

    
