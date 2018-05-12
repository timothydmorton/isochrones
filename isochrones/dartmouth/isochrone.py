from __future__ import division,print_function
import os,os.path, glob, re
import numpy as np
from pkg_resources import resource_filename
import logging

from scipy.interpolate import LinearNDInterpolator as interpnd

# Check to see if building on ReadTheDocs
from ..config import on_rtd

if not on_rtd:
    import pandas as pd
else:
    pd = None

import pickle

from ..isochrone import Isochrone, FastIsochrone
from ..config import ISOCHRONES

TRI_FILE = '{}/dartmouth.tri'.format(ISOCHRONES)

if not on_rtd:
    MAXAGES = np.load(resource_filename('isochrones','data/dartmouth_maxages.npz'))
    MAXAGE = interpnd(MAXAGES['points'], MAXAGES['maxages'])

from .grid import DartmouthModelGrid

TRI = None

class Dartmouth_Isochrone(Isochrone):
    """Dotter (2008) Stellar Models, at solar a/Fe and He abundances.

    :param bands: (optional)
        List of desired photometric bands.  Default list of bands is
        ``['B','V','g','r','i','z','J','H','K','W1','W2','W3','Kepler']``.

    Model grids are obtained from `here <http://stellar.dartmouth.edu/models/>`_
    """
    name = 'dartmouth'
    default_bands = DartmouthModelGrid.default_bands

    def __init__(self,bands=None,
                 afe='afep0', y='', **kwargs): # minage=9 removed
        if bands is None:
            bands = list(self.default_bands)

        if afe != 'afep0' and y != '':
            raise NotImplementedError('Model grids not prepared for non-solar [alpha/Fe] or y')

        df = DartmouthModelGrid(bands, afe=afe, y=y).df
        # df = get_grid(bands, afe=afe, y=y)

        global TRI

        if TRI is None:
            DartmouthModelGrid.verify_grids()
            try:
                f = open(TRI_FILE,'rb')
                TRI = pickle.load(f)
            except:
                f = open(TRI_FILE,'rb')
                TRI = pickle.load(f,encoding='latin-1')
            finally:
                f.close()


        mags = {b:df[b].values for b in bands}

        Isochrone.__init__(self,df['MMo'].values, df['age'].values,
                           df['feh'].values,df['MMo'].values, df['LogLLo'].values,
                           10**df['LogTeff'].values,df['LogG'].values,mags,tri=TRI,
                           **kwargs)

    def agerange(self, m, feh=0.0):
        minage = self.minage * np.ones_like(m)
        maxage = MAXAGE(m, feh) * np.ones_like(m)
        return minage,maxage


class Dartmouth_FastIsochrone(FastIsochrone):
    name = 'dartmouth'
    age_col = 5
    feh_col = 6
    mass_col = 1
    loggTeff_col = 2
    logg_col = 3
    logL_col = 4
    modelgrid = DartmouthModelGrid
    default_bands = DartmouthModelGrid.default_bands

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
