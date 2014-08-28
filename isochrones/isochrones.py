from __future__ import division,print_function
import numpy as np
import os,sys,re,os.path
from scipy.interpolate import LinearNDInterpolator as interpnd
import scipy.optimize
import numpy.random as rand

from astropy import constants as const

G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value

import pandas as pd

class Isochrone(object):
    """Generic 2d isochrone class. Only valid for single metallicity.

    Main functionality is interpolation functions that return M, R, mags, etc.
    for given values of mass and age.  Thus, this class is theoretically oriented:
    to generate stellar models for given mass and age.  It is not optimized to fit
    observed Teff, fe/H, logg., because of fixed fe/H.  
    """
    def __init__(self,age,m_ini,m_act,logL,Teff,logg,mags):
        """if feh is included, becomes 3d, and unweildy...
        """
        self.is3d = False #generic 3-d isochrone not implemented

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
            return np.sqrt(G*self.M(*pts)*MSUN/10**self.logg(*pts))/RSUN
        self.R = R_fn

        self.mag = {band:interpnd(self.tri,mags[band]) for band in self.bands}
        #for band in self.bands:
        #    self.mag[band] = interpnd(points,mags[band])

    def __call__(self,*args):
        m,age = args #need to change if 3d implemented
        Ms = self.M(*args)
        Rs = self.R(*args)
        logLs = self.logL(*args)
        loggs = self.logg(*args)
        Teffs = self.Teff(*args)
        mags = {band:self.mag[band](*args) for band in self.bands}
        
        #feh left out of this; need to put back if 3d implemented
        return {'age':age,'M':Ms,'R':Rs,'logL':logLs,'logg':loggs,'Teff':Teffs,'mag':mags}        

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
        
def isofit(iso,p0=None,**kwargs):
    """Finds best leastsq match to provided (val,err) keyword pairs.

    e.g. isofit(iso,Teff=(5750,50),logg=(4.5,0.1))
    """
    def chisqfn(pars):
        tot = 0
        for kw in kwargs:
            val,err = kwargs[kw]
            fn = getattr(iso,kw)
            tot += (val-fn(*pars))**2/err**2
        return tot
    if iso.is3d:
        if p0 is None:
            p0 = ((iso.minm+iso.maxm)/2,(iso.minage + iso.maxage)/2.,(iso.minfeh + iso.maxfeh)/2.)
    else:
        if p0 is None:
            p0 = (1,9.5)
    pfit = scipy.optimize.fmin(chisqfn,p0,disp=False)
    return iso(*pfit)

def shotgun_isofit(iso,n=100,**kwargs):
    """Rudimentarily finds distribution of best-fits by finding leastsq match to MC sample of points
    """
    simdata = {}
    for kw in kwargs:
        val,err = kwargs[kw]
        simdata[kw] = rand.normal(size=n)*err + val
    if iso.is3d:
        Ms,ages,fehs = (np.zeros(n),np.zeros(n),np.zeros(n))
    else:
        Ms,ages = (np.zeros(n),np.zeros(n))
    for i in np.arange(n):
        simkwargs = {}
        for kw in kwargs:
            val = simdata[kw][i]
            err = kwargs[kw][1]
            simkwargs[kw] = (val,err)
        fit = isofit(iso,**simkwargs)
        Ms[i] = fit['M']
        ages[i] = fit['age']
        if iso.is3d:
            fehs[i] = fit['feh']

    if iso.is3d:
        res = iso(Ms,ages,fehs)
    else:
        res = iso(Ms,ages)
    return res

        
def fehstr(feh,minfeh=-1.0,maxfeh=0.5):
        if feh < minfeh:
            return '%.1f' % minfeh
        elif feh > maxfeh:
            return '%.1f' % maxfeh
        elif (feh > -0.05 and feh < 0):
            return '0.0'
        else:
            return '%.1f' % feh            
