from __future__ import division,print_function
import os,os.path
import numpy as np
import pkg_resources

import pandas as pd
import pickle

from . import isochrones as iso

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

MASTERFILE = '{}/padova.h5'.format(DATADIR)
MASTERDF = pd.read_hdf(MASTERFILE,'df')
TRI_FILE = '{}/padova.tri'.format(DATADIR)

class Padova_Isochrone(iso.Isochrone):
    """Padova Stellar Models
    """
    def __init__(self,bands=['bol','Kepler','g','r','i',
                             'z','D51','J','H','K']):

        df = MASTERDF

        mags = {}
        for band in bands:
            mags[band] = df[band]

        f = open(TRI_FILE,'rb')
        tri = pickle.load(f)
        f.close()
        
        iso.Isochrone.__init__(self,df['M_ini'],df['age'],df['feh'],
                               df['M_act'],df['logL'],
                               10**df['logTeff'],df['logg'],mags,tri=tri)

    
