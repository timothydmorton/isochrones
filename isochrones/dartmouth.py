from __future__ import division,print_function
import os,os.path
import numpy as np
import pkg_resources

from scipy.interpolate import LinearNDInterpolator as interpnd
import pandas as pd
import pickle

from .isochrones import Isochrone

#DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
DATADIR = os.path.expanduser('~/.isochrones')
if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)

MASTERFILE = '{}/dartmouth.h5'.format(DATADIR)
TRI_FILE = '{}/dartmouth.tri'.format(DATADIR)

def _download_h5():
    #url = 'http://files.figshare.com/1801331/dartmouth.h5'
    url = 'http://zenodo.org/record/12800/files/dartmouth.h5'
    import urllib
    print('Downloading Dartmouth stellar model data (should happen only once)...')
    if os.path.exists(MASTERFILE):
        os.remove(MASTERFILE)
    urllib.urlretrieve(url,MASTERFILE)

def _download_tri():
    url = 'http://zenodo.org/record/12800/files/dartmouth.tri'
    #url = 'http://files.figshare.com/1801343/dartmouth.tri'
    import urllib
    print('Downloading Dartmouth isochrone pre-computed triangulation (should happen only once...)')
    if os.path.exists(TRI_FILE):
        os.remove(TRI_FILE)
    urllib.urlretrieve(url,TRI_FILE)

if not os.path.exists(MASTERFILE):
    _download_h5()

if not os.path.exists(TRI_FILE):
    _download_tri()

MASTERDF = pd.read_hdf(MASTERFILE,'df')

class Dartmouth_Isochrone(Isochrone):
    """Dotter (2008) Stellar Models
    """
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
        
        Isochrone.__init__(self,df['M'],np.log10(df['age']*1e9),
                           df['feh'],df['M'],df['logL'],
                           10**df['logTeff'],df['logg'],mags,tri=tri)


############ utility functions used to set up data sets from original isochrone data files ########

DARTMOUTH_DATAFOLDER = ''  #this would be location of raw data files.

def write_tri(df=MASTERDF, outfile=TRI_FILE):
    """Writes the Delanuay triangulation of the models to file.  Takes a few minutes, so beware.

    Typically no need to do this unless you can't download the .tri file for some reason...
    """
    N = len(df)
    pts = np.zeros((N,3))
    pts[:,0] = np.array(df['M'])
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
