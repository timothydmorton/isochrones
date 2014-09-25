from __future__ import division,print_function
import os,os.path
import numpy as np
import pkg_resources

import pandas as pd
import pickle

from . import isochrones as iso

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

MASTERFILE = '{}/basti.h5'.format(DATADIR)
MASTERDF = pd.read_hdf(MASTERFILE,'df')
TRI_FILE = '{}/basti.tri'.format(DATADIR)

from astropy import constants as const
RSUN = const.R_sun.cgs.value
SIGMA = const.sigma_sb.cgs.value

class Basti_Isochrone(iso.Isochrone):
    """BASTI stellar models

    http://basti.oa-teramo.inaf.it/index.html, version 5.0.1
    """
    def __init__(self):
        df = MASTERDF

        f = open(TRI_FILE,'rb')
        tri = pickle.load(f)
        f.close()

        mags = {}

        logL = np.log10(4*np.pi*(df['radius']*RSUN)**2 * SIGMA * (10**(df['logTeff']))**4)

        iso.Isochrone.__init__(self,df['mini'],df['logage'],df['feh'],df['mass'],
                               logL,10**df['logTeff'],df['logg'],mags,tri=tri)


