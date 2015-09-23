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

MASTERFILE = '{}/padova.h5'.format(DATADIR)
TRI_FILE = '{}/padova.tri'.format(DATADIR)

def _download_h5():
    url = 'http://zenodo.org/record/12800/files/padova.h5'
    from six.moves import urllib
    print('Downloading Padova stellar model data (should happen only once)...')
    if os.path.exists(MASTERFILE):
        os.remove(MASTERFILE)
    urllib.request.urlretrieve(url,MASTERFILE)

def _download_tri():
    url = 'http://zenodo.org/record/12800/files/padova.tri'
    from six.moves import urllib
    print('Downloading Padova isochrone pre-computed triangulation (should happen only once...)')
    if os.path.exists(TRI_FILE):
        os.remove(TRI_FILE)
    urllib.request.urlretrieve(url,TRI_FILE)

if not os.path.exists(MASTERFILE):
    _download_h5()

if not os.path.exists(TRI_FILE):
    _download_tri()

MASTERDF = pd.read_hdf(MASTERFILE,'df')


class Padova_Isochrone(Isochrone):
    """Padova Stellar Models
    """
    def __init__(self,bands=['bol','Kepler','g','r','i',
                             'z','D51','J','H','K']):

        df = MASTERDF

        mags = {}
        for band in bands:
            mags[band] = df[band]

        try:
            f = open(TRI_FILE,'rb')
            tri = pickle.load(f)
        except:
            f = open(TRI_FILE,'rb')
            tri = pickle.load(f,encoding='latin-1')
        finally:
            f.close()
        
        Isochrone.__init__(self,df['M_ini'],df['age'],df['feh'],
                           df['M_act'],df['logL'],
                           10**df['logTeff'],df['logg'],mags,tri=tri)

    
######## setup functions.  shouldn't have to use ever.

DATAFOLDER = '' #datafolder where raw padova tracks would live in order to recreate dataframe, triangulation, etc.

def fehstr(feh,minfeh=-2.0,maxfeh=0.2):
        if feh < minfeh:
            return '%.1f' % minfeh
        elif feh > maxfeh:
            return '%.1f' % maxfeh
        elif (feh > -0.05 and feh < 0):
            return '0.0'
        else:
            return '%.1f' % feh            


def padova_to_df(save=True,savefile='isochrones/data/padova.h5'):
    alldf = pd.DataFrame()
    
    for feh in np.arange(-2,0.21,0.1):
        filename = DATAFOLDER + '/padova_%s.dat' % fehstr(feh,-2,0.2)    
        age,m_ini,m_act,logL,logT,logg,mbol,kep,g,r,i,z,dd051,J,H,K = \
                np.loadtxt(filename,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),unpack=True)
        df = pd.DataFrame()
        df['age'] = age
        df['M_ini'] = m_ini
        df['M_act'] = m_act
        df['logL'] = logL
        df['logTeff'] = logT
        df['logg'] = logg
        df['bol'] = mbol
        df['Kepler'] = kep
        df['g'] = g
        df['r'] = r
        df['i'] = i
        df['z'] = z
        df['D51'] = dd051
        df['J'] = J
        df['H'] = H
        df['K'] = K
        df['feh'] = feh * np.ones_like(age)
        alldf = alldf.append(df)

    if save:
        alldf.to_hdf(savefile,'df')
        
    return alldf

def write_tri(df=MASTERDF,outfile=TRI_FILE):
        N = len(df)
        pts = np.zeros((N,3))
        pts[:,0] = np.array(df['M_ini'])
        pts[:,1] = np.array(df['age'])
        pts[:,2] = np.array(df['feh'])
        Jmags = np.array(df['J'])

        Jfn = interpnd(pts,Jmags)

        f = open(outfile,'wb')
        pickle.dump(Jfn.tri,f)
        f.close()
