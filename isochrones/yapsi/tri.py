from ..config import ISOCHRONES
import sys, os
import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.interpolate import LinearNDInterpolator as interpnd

from .grid import YAPSIModelGrid

def write_tri(filename=os.path.join(ISOCHRONES,'yapsi.tri')):
    df = YAPSIModelGrid(['V']).df
    N = len(df)
    pts = np.zeros((N,3))
    pts[:,0] = np.array(df['mass'])
    pts[:,1] = np.array(df['age'])
    pts[:,2] = np.array(df['feh'])
    mags = np.array(df['V'])

    fn = interpnd(pts,mags)

    with open(filename,'wb') as f:
        pickle.dump(fn.tri,f)
