import os,sys,os.path
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator as interpnd
import pickle

from astropy import constants as const
G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value

#TABLEFILE = 'isochrones/data/grid_basti_m.0.05_f.0.1.dat'
TABLEFILE = 'isochrones/data/grid_basti_original.dat'
#H5FILE = 'isochrones/data/basti_interpolated.h5'
H5FILE = 'isochrones/data/basti.h5'

if os.path.exists(H5FILE):
    DF = pd.read_hdf(H5FILE,'df')
else:
    DF = pd.read_table(TABLEFILE,
                    names=['mini','feh','mass',
                            'logTeff','radius','logage'],
                            delim_whitespace=True)


    DF['logg'] = np.log10(G*DF['mass']*MSUN/(DF['radius']*RSUN)**2)
    DF.to_hdf(H5FILE,'df')
    

def write_tri(df=DF,outfile='isochrones/data/basti.tri'):
        N = len(df)
        pts = np.zeros((N,3))
        pts[:,0] = np.array(df['mini'])
        pts[:,1] = np.array(df['logage'])
        pts[:,2] = np.array(df['feh'])

        Rs = np.array(df['radius'])

        Rfn = interpnd(pts,Rs)

        f = open(outfile,'wb')
        pickle.dump(Rfn.tri,f)
        f.close()
        
