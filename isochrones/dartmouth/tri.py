#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import cPickle as pickle
from scipy.interpolate import LinearNDInterpolator as interpnd

from ..config import ISOCHRONES
DATADIR = os.path.join(ISOCHRONES, 'dartmouth')

from .grid import DartmouthModelGrid

def write_tri(filename=os.path.join(DATADIR,'dartmouth.tri')):
    df = DartmouthModelGrid(['g'])
    N = len(df)
    pts = np.zeros((N,3))
    pts[:,0] = np.array(df['MMo'])
    pts[:,1] = df['age']
    pts[:,2] = np.array(df['feh'])
    gmags = np.array(df['sdss_g'])

    gfn = interpnd(pts,gmags)

    with open(filename,'wb') as f:
        pickle.dump(gfn.tri,f)
