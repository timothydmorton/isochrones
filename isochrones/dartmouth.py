from __future__ import division,print_function
import os,os.path, glob, re
import numpy as np
from pkg_resources import resource_filename
import logging

from scipy.interpolate import LinearNDInterpolator as interpnd
try:
    import pandas as pd
except ImportError:
    pd = None
    
import pickle

from .isochrone import Isochrone

#DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
DATADIR = os.getenv('ISOCHRONES',
                    os.path.expanduser(os.path.join('~','.isochrones')))
if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)

MASTERFILE = '{}/dartmouth.h5'.format(DATADIR)
TRI_FILE = '{}/dartmouth.tri'.format(DATADIR)

MAXAGES = np.load(resource_filename('isochrones','data/dartmouth_maxages.npz'))
MAXAGE = interpnd(MAXAGES['points'], MAXAGES['maxages'])

# Columns in DataFrames that are *not* magnitudes
COMMON_COLUMNS = ['EEP', 'MMo', 'LogTeff', 'LogG', 'LogLLo', 'age', 'feh']

# Default available photometric systems
PHOT_SYSTEMS = ['SDSSugriz','UBVRIJHKsKp','WISE','LSST','UKIDSS']

# Names of bands in each Dartmouth photometric system grid
PHOT_BANDS = dict(SDSSugriz=['sdss_z', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_g'],
                  UBVRIJHKsKp=['B', 'I', 'H', 'J', 'Ks', 'R', 'U', 'V', 'D51', 'Kp'],
                  WISE=['W4', 'W3', 'W2', 'W1'],
                  LSST=['LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'LSST_g', 'LSST_i'],
                  UKIDSS=['Y', 'H', 'K', 'J', 'Z'])

def _band_name(b):
    """Defines what a "shortcut" band name refers to.  Returns phot_system, band
    """
    # Default to SDSS for these

    if b in ['u','g','r','i','z']:
        sys = 'SDSSugriz'
        band = 'sdss_{}'.format(b)
    elif b in ['U','B','V','R','I','J','H','Ks']:
        sys = 'UBVRIJHKsKp'
        band = b
    elif b=='K':
        sys = 'UBVRIJHKsKp'
        band = 'Ks'
    elif b in ['kep','Kepler','Kp']:
        sys = 'UBVRIJHKsKp'
        band = 'Kp'
    elif b in ['W1','W2','W3','W4']:
        sys = 'WISE'
        band = b
    else:
        m = re.match('(\w+)_([a-zA-Z_]+)',b)
        if m:
            if m.group(1) in PHOT_SYSTEMS:
                sys = m.group(1)
                band = m.group(2)
            elif m.group(1)=='UK':
                sys = 'UKIDSS'
                band = m.group(2)
    return sys, band


def _download_h5():
    """
    Downloads HDF5 file containing Dartmouth grids from Zenodo.
    """
    #url = 'http://zenodo.org/record/12800/files/dartmouth.h5'
    url = 'http://zenodo.org/record/15843/files/dartmouth.h5'
    from six.moves import urllib
    print('Downloading Dartmouth stellar model data (should happen only once)...')
    if os.path.exists(MASTERFILE):
        os.remove(MASTERFILE)
    urllib.request.urlretrieve(url,MASTERFILE)

def _download_tri():
    """
    Downloads pre-computed triangulation for Dartmouth grids from Zenodo.
    """
    #url = 'http://zenodo.org/record/12800/files/dartmouth.tri'
    #url = 'http://zenodo.org/record/15843/files/dartmouth.tri'
    url = 'http://zenodo.org/record/17627/files/dartmouth.tri'
    from six.moves import urllib
    print('Downloading Dartmouth isochrone pre-computed triangulation (should happen only once...)')
    if os.path.exists(TRI_FILE):
        os.remove(TRI_FILE)
    urllib.request.urlretrieve(url,TRI_FILE)

if not os.path.exists(MASTERFILE):
    _download_h5()

if not os.path.exists(TRI_FILE):
    _download_tri()

#Check to see if you have the right dataframe and tri file
import hashlib

DF_SHASUM = '0515e83521f03cfe3ab8bafcb9c8187a90fd50c7'
TRI_SHASUM = 'e05a06c799abae3d526ac83ceeea5e6df691a16d'

if hashlib.sha1(open(MASTERFILE, 'rb').read()).hexdigest() != DF_SHASUM:
    raise ImportError('You have a wrong/corrupted/outdated Dartmouth DataFrame!' + 
                      ' Delete {} and try re-importing to download afresh.'.format(MASTERFILE))
if hashlib.sha1(open(TRI_FILE, 'rb').read()).hexdigest() != TRI_SHASUM:
    raise ImportError('You have a wrong/corrupted/outdated Dartmouth triangulation!' + 
                      ' Delete {} and try re-importing to download afresh.'.format(TRI_FILE))

#


if pd is not None:
    MASTERDF = pd.read_hdf(MASTERFILE,'df').dropna() #temporary hack
else:
    MASTERDF = None
    
class Dartmouth_Isochrone(Isochrone):
    """Dotter (2008) Stellar Models, at solar a/Fe and He abundances.

    :param bands: (optional)
        List of desired photometric bands.  Must be a subset of
        ``['U','B','V','R','I','J','H','K','g','r','i','z','Kepler','D51',
        'W1','W2','W3']``, which is the default.  W4 is not included
        because it does not have a well-measured A(lambda)/A(V).


    """
    def __init__(self,bands=['U','B','V','R','I','J','H',
                             'K','g','r','i','z','Kepler','D51',
                             'W1','W2','W3'], minage=9, **kwargs):

        df = MASTERDF

        mags = {}
        for band in bands:
            try:
                if band in ['g','r','i','z']:
                    mags[band] = df['sdss_{}'.format(band)]
                else:
                    mags[band] = df[band]
            except:
                if band == 'kep' or band == 'Kepler':
                    mags[band] = df['Kp']
                elif band == 'K':
                    mags['K'] = df['Ks']
                else:
                    raise

        try:
            f = open(TRI_FILE,'rb')
            tri = pickle.load(f)
        except:
            f = open(TRI_FILE,'rb')
            tri = pickle.load(f,encoding='latin-1')
        finally:
            f.close()
        
        Isochrone.__init__(self,df['M/Mo'],np.log10(df['age']*1e9),
                           df['feh'],df['M/Mo'],df['LogL/Lo'],
                           10**df['LogTeff'],df['LogG'],mags,tri=tri, 
                           minage=minage, **kwargs)

    def agerange(self, m, feh=0.0):
        minage = self.minage * np.ones_like(m)
        maxage = MAXAGE(m, feh) * np.ones_like(m)
        return minage,maxage
        

############ utility functions used to set up data sets from original isochrone data files----these are obselete, I believe, now! ########

DARTMOUTH_DATAFOLDER = ''  #this would be location of raw data files.

def write_tri(df=MASTERDF, outfile=TRI_FILE):
    """Writes the Delanuay triangulation of the models to file.  Takes a few minutes, so beware.

    Typically no need to do this unless you can't download the .tri file for some reason...
    """
    N = len(df)
    pts = np.zeros((N,3))
    pts[:,0] = np.array(df['M/Mo'])
    pts[:,1] = np.log10(np.array(df['age'])*1e9)
    pts[:,2] = np.array(df['feh'])
    Jmags = np.array(df['J'])

    Jfn = interpnd(pts,Jmags)

    f = open(outfile,'wb')
    pickle.dump(Jfn.tri,f)
    f.close()

    
def fehstr(feh,minfeh=-1.0,maxfeh=0.5):
        if feh < minfeh:
            return '%.1f' % minfeh
        elif feh > maxfeh:
            return '%.1f' % maxfeh
        elif (feh > -0.05 and feh < 0):
            return '0.0'
        else:
            return '%.1f' % feh            

def writeall_h5(fehs=np.arange(-1,0.51,0.1)):
    store = pd.HDFStore('%s/dartmouth.h5' % DARTMOUTH_DATAFOLDER)
    for feh in fehs:
        name = fehstr(feh)
        print(name)
        store[name] = dartmouth_to_df(feh)
    store.close()

def dartmouth_all_df(fehs=np.arange(-1,0.51,0.1),savefile=None):
    store = pd.HDFStore('%s/dartmouth.h5' % DARTMOUTH_DATAFOLDER)
    df = pd.DataFrame()
    for feh in fehs:
        newdf = store[fehstr(feh)].copy()
        newdf['feh'] = np.ones(len(newdf))*feh
        df = df.append(newdf)
    store.close()
    if savefile is not None:
        df.to_hdf(savefile)
    return df

def dartmouth_to_df(feh):
    filename_2mass = '%s/dartmouth_%s_2massKp.iso' % (DARTMOUTH_DATAFOLDER,fehstr(feh,-1.0,0.5))
    filename_ugriz = '%s/dartmouth_%s_ugriz.iso' % (DARTMOUTH_DATAFOLDER,fehstr(feh,-1.0,0.5))    

    bands_ugriz = ['u','g','r','i','z']
    darnames_ugriz = ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']

    bands_2mass = ['U','B','V','R','I','J','H','Ks','Kp','D51']
    darnames_2mass = bands_2mass

    rec = np.recfromtxt(filename_2mass,skiprows=8,names=True)
    rec2 = np.recfromtxt(filename_ugriz,skiprows=8,names=True,usecols=(5,6,7,8,9))
    df = pd.DataFrame(rec)
    df2 = pd.DataFrame(rec2)
    for b,d in zip(bands_ugriz,darnames_ugriz):
        df[b] = df2[d]

    n = len(df)
    ages = np.zeros(n)
    curage = 0
    i=0
    for line in open(filename_2mass):
        m = re.match('#',line)
        if m:
            m = re.match('#AGE=\s*(\d+\.\d+)\s+',line)
            if m:
                curage=float(m.group(1))
        else:
            if re.search('\d',line):
                ages[i]=curage
                i+=1

    df['age'] = ages
    columns = {darnames_2mass[i]:bands_2mass[i] for i in range(len(bands_2mass))}
    columns.update({'MMo':'M','LogTeff':'logTeff',
                    'LogG':'logg','LogLLo':'logL'})
    df.rename(columns=columns,inplace=True)
    return df

############# downloading files from stellar.dartmouth.edu

def download_evtracks(fehs=[-2.5,-2.0,-1.5,-1.0,-0.5,0.0, 0.15, 0.3, 0.5],
                      afe=0., phot_system='sdss'):
    import urllib

    urlbase = 'http://stellar.dartmouth.edu/models/tracks/{}/'.format(phot_system)

    for feh in fehs:
        print('Fetching evolution tracks for feh={}...'.format(feh))
        feh_sign = 'p' if feh >= 0 else 'm'
        afe_sign = 'p' if afe >= 0 else 'm'
        filename = 'feh{}{:02.0f}afe{}{:01.0f}_{}.tgz'.format(feh_sign,abs(feh*10),
                                                            afe_sign,abs(afe*10),
                                                            phot_system)
        url = urlbase+filename

        folder = os.path.join(DATADIR, 'dartmouth')
        if not os.path.exists(folder):
            os.makedirs(folder)
        localfile = os.path.join(folder,filename)
        if not os.path.exists(localfile):
            urllib.urlretrieve(url,localfile)

def write_maxages(fehs=[-2.5,-2.0,-1.5,-1.0,-0.5,0.0, 0.15, 0.3, 0.5],
                  afe=0., phot_system='sdss', savefile='maxages.npz'):
    m_list = []
    feh_list = []
    maxage_list = []
    for feh in fehs:
        feh_sign = 'p' if feh >= 0 else 'm'
        afe_sign = 'p' if afe >= 0 else 'm'
        name = 'feh{}{:02.0f}afe{}{:01.0f}'.format(feh_sign,abs(feh*10),
                                                            afe_sign,abs(afe*10))
                                
        folder = os.path.join(DATADIR,'dartmouth',name)
        files = glob.glob('{}/m*'.format(folder))
        for file in files:
            m = re.search('m(\d\d\d)feh',file)
            if m:
                mass = int(m.group(1))/100.
            ages = np.loadtxt(file, usecols=(0,))
            feh_list.append(feh)
            m_list.append(mass)
            maxage_list.append(np.log10(ages[-1]))

    points = np.array([m_list, feh_list]).T
    maxages = np.array(maxage_list)
    np.savez(savefile, points=points, maxages=maxages)
