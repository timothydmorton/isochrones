from __future__ import division,print_function
import os,os.path, glob, re
import numpy as np
from pkg_resources import resource_filename
import logging

from scipy.interpolate import LinearNDInterpolator as interpnd

# Check to see if building on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    import pandas as pd
else:
    pd = None
    
import pickle

from ..isochrone_new import Isochrone
from ..config import ISOCHRONES

TRI_FILE = '{}/dartmouth.tri'.format(ISOCHRONES)

if not on_rtd:
    MAXAGES = np.load(resource_filename('isochrones','data/dartmouth_maxages.npz'))
    MAXAGE = interpnd(MAXAGES['points'], MAXAGES['maxages'])

    # Download data if you need to.
    if not os.path.exists(TRI_FILE):
        from .fileutils import download_grids
        download_grids()

    #Check to see if you have the right tri file
    import hashlib
    if hashlib.md5(open(TRI_FILE,'rb').read()).hexdigest() != '477f5b835c0e805810a3154922eeb3d6':
        raise ImportError('You have a wrong/corrupted/outdated Dartmouth triangulation!' + 
                          ' Delete {} and try re-importing to download afresh.'.format(TRI_FILE))


from .grids import DartmouthModelGrid

TRI = None

class Dartmouth_Isochrone(Isochrone):
    """Dotter (2008) Stellar Models, at solar a/Fe and He abundances.

    :param bands: (optional)
        List of desired photometric bands. 

    """
    def __init__(self,bands=['B','V','g','r','i','z',
                             'J','H','K',
                             'W1','W2','W3','Kepler'], 
                 afe='afep0', y='', **kwargs): # minage=9 removed
        if afe != 'afep0' and y != '':
            raise NotImplementedError('Model grids not prepared for non-solar [alpha/Fe] or y')

        df = DartmouthModelGrid(bands, afe=afe, y=y).df
        # df = get_grid(bands, afe=afe, y=y)

        global TRI

        if TRI is None:
            try:
                f = open(TRI_FILE,'rb')
                TRI = pickle.load(f)
            except:
                f = open(TRI_FILE,'rb')
                TRI = pickle.load(f,encoding='latin-1')
            finally:
                f.close()

        
        mags = {b:df[b].values for b in bands}

        Isochrone.__init__(self,df['MMo'].values,np.log10(df['age'].values*1e9),
                           df['feh'].values,df['MMo'].values,df['LogLLo'].values,
                           10**df['LogTeff'].values,df['LogG'].values,mags,tri=TRI, 
                           **kwargs)

    def agerange(self, m, feh=0.0):
        minage = self.minage * np.ones_like(m)
        maxage = MAXAGE(m, feh) * np.ones_like(m)
        return minage,maxage
        

#### Old utility function.  this needs to be updated.

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
