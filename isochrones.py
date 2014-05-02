from __future__ import division,print_function
import numpy as np
import os,sys,re,os.path
from scipy.interpolate import LinearNDInterpolator as interpnd

import pandas as pd

class Isochrone(object):
    """Generic 2d isochrone class. Only valid for single metallicity.
    """
    def __init__(self,age,m_ini,m_act,logL,Teff,logg,mags):
        """if feh is included, becomes 3d, and unweildy...
        """
        self.minage = age.min()
        self.maxage = age.max()
        self.minmass = m_act.min()
        self.maxmass = m_act.max()
        
        self.bands = []
        for band in mags:
            self.bands.append(band)

        L = 10**logL

        points = np.zeros((len(m_ini),2))
        points[:,0] = m_ini
        points[:,1] = age

        self.M = interpnd(points,m_act)

        self.tri = self.M.tri

        self.logL = interpnd(self.tri,logL)
        self.logg = interpnd(self.tri,logg)
        self.logTeff = interpnd(self.tri,np.log10(Teff))
        def Teff_fn(*pts):
            return 10**self.logTeff(*pts)

        self.Teff = Teff_fn
        def R_fn(*pts):
            return sqrt(G*self.M(*pts)*MSUN/10**self.logg(*pts))/RSUN
        self.R = R_fn

        self.mag = {band:interpnd(self.tri,mags[band]) for band in self.bands}
        #for band in self.bands:
        #    self.mag[band] = interpnd(points,mags[band])

    def __call__(self,*args):
        
        Ms = self.M(*args)
        Rs = self.R(*args)
        logLs = self.logL(*args)
        loggs = self.logg(*args)
        Teffs = self.Teff(*args)
        mags = {band:self.mag[band](*args) for band in self.bands}
        #for band in self.bands:
        #    mags[band] = self.mag[band](*args)
        return {'age':age,'M':Ms,'feh':self.feh(*args),'R':Rs,'logL':logLs,'logg':loggs,'Teff':Teffs,'mag':mags}        

    def evtrack(self,m,minage=6.7,maxage=10,dage=0.05):
        ages = np.arange(minage,maxage,dage)
        Ms = self.M(m,ages)
        Rs = self.R(m,ages)
        logLs = self.logL(m,ages)
        loggs = self.logg(m,ages)
        Teffs = self.Teff(m,ages)
        mags = {band:self.mag[band](m,ages) for band in self.bands}
        #for band in self.bands:
        #    mags[band] = self.mag[band](m,ages)

        #return array([ages,Ms,Rs,logLs,loggs,Teffs,   #record array?
        return {'age':ages,'M':Ms,'R':Rs,'logL':logLs,'Teff':Teffs,'mag':mags}
            
    def isochrone(self,age,minm=0.1,maxm=2,dm=0.02):
        ms = np.arange(minm,maxm,dm)
        ages = np.ones(ms.shape)*age

        Ms = self.M(ms,ages)
        Rs = self.R(ms,ages)
        logLs = self.logL(ms,ages)
        loggs = self.logg(ms,ages)
        Teffs = self.Teff(ms,ages)
        mags = {band:self.mag[band](ms,ages) for band in self.bands}
        #for band in self.bands:
        #    mags[band] = self.mag[band](ms,ages)

        return {'M':Ms,'R':Rs,'logL':logLs,'Teff':Teffs,'mag':mags}        
        
