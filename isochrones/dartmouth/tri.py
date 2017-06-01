#!/usr/bin/env python

import sys, os
import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.interpolate import LinearNDInterpolator as interpnd

from ..config import ISOCHRONES

from .grid import DartmouthModelGrid

def write_tri(filename=os.path.join(ISOCHRONES,'dartmouth.tri')):
    df = DartmouthModelGrid(['g']).df
    N = len(df)
    pts = np.zeros((N,3))
    pts[:,0] = np.array(df['MMo'])
    pts[:,1] = np.array(df['age'])
    pts[:,2] = np.array(df['feh'])
    gmags = np.array(df['g'])

    gfn = interpnd(pts,gmags)

    with open(filename,'wb') as f:
        pickle.dump(gfn.tri,f)
