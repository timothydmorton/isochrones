from __future__ import division,print_function
import os,os.path
import numpy as np
import pkg_resources

from scipy.interpolate import LinearNDInterpolator as interpnd
import pandas as pd
import pickle

from .isochrone import Isochrone

#DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
DATADIR = os.path.expanduser('~/.isochrones')
if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)

MASTERFILE = '{}/basti.h5'.format(DATADIR)
TRI_FILE = '{}/basti.tri'.format(DATADIR)

def _download_h5():
    url = 'http://zenodo.org/record/12800/files/basti.h5'
    from six.moves import urllib
    print('Downloading BASTI stellar model data (should happen only once)...')
    if os.path.exists(MASTERFILE):
        os.remove(MASTERFILE)
    urllib.request.urlretrieve(url,MASTERFILE)

def _download_tri():
    url = 'http://zenodo.org/record/12800/files/basti.tri'
    from six.moves import urllib
    print('Downloading BASTI isochrone pre-computed triangulation (should happen only once...)')
    if os.path.exists(TRI_FILE):
        os.remove(TRI_FILE)
    urllib.request.urlretrieve(url,TRI_FILE)

if not os.path.exists(MASTERFILE):
    _download_h5()

if not os.path.exists(TRI_FILE):
    _download_tri()


MASTERDF = pd.read_hdf(MASTERFILE,'df')


from astropy import constants as const
RSUN = const.R_sun.cgs.value
SIGMA = const.sigma_sb.cgs.value

class Basti_Isochrone(Isochrone):
    """BASTI stellar models

    http://basti.oa-teramo.inaf.it/index.html, version 5.0.1
    """
    def __init__(self):
        df = MASTERDF

        try:
            f = open(TRI_FILE,'rb')
            tri = pickle.load(f)
        except:
            f = open(TRI_FILE,'rb')
            tri = pickle.load(f,encoding='latin-1')
        finally:
            f.close()

        mags = {}

        logL = np.log10(4*np.pi*(df['radius']*RSUN)**2 * SIGMA * (10**(df['logTeff']))**4)

        Isochrone.__init__(self,df['mini'],df['logage'],df['feh'],df['mass'],
                           logL,10**df['logTeff'],df['logg'],mags,tri=tri)


############################
# the below is just for setup purposes and should never really be used....
if False:
    DATAFOLDER = ''

    #TABLEFILE = 'isochrones/data/grid_basti_m.0.05_f.0.1.dat'
    TABLEFILE = '{}/grid_basti_original.dat'.format(DATAFOLDER)
    #H5FILE = 'isochrones/data/basti_interpolated.h5'
    H5FILE = '{}/basti.h5'.format(DATAFOLDER)

    if os.path.exists(H5FILE):
        DF = pd.read_hdf(H5FILE,'df')
    else:
        DF = pd.read_table(TABLEFILE,
                        names=['mini','feh','mass',
                                'logTeff','radius','logage'],
                                delim_whitespace=True)


        DF['logg'] = np.log10(G*DF['mass']*MSUN/(DF['radius']*RSUN)**2)
        DF.to_hdf(H5FILE,'df')
    

def write_tri(df=MASTERDF,outfile=TRI_FILE):
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
        
