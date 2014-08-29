from __future__ import division,print_function
import numpy as np
import os,sys,re,os.path
from scipy.interpolate import LinearNDInterpolator as interpnd
import scipy.optimize
import numpy.random as rand
import emcee

from astropy import constants as const

G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

EXTINCTIONFILE = '{}/extinction.txt'.format(DATADIR)
EXTINCTION = dict()
EXTINCTION5 = dict()
for line in open(EXTINCTIONFILE,'r'):
    line = line.split()
    EXTINCTION[line[0]] = float(line[1])
    EXTINCTION5[line[0]] = float(line[2])

EXTINCTION['kep'] = 0.85946
EXTINCTION['V'] = 1.0
EXTINCTION['Ks'] = EXTINCTION['K']
EXTINCTION['Kepler'] = EXTINCTION['kep']


import pandas as pd

class Isochrone(object):
    """Generic isochrone class. Everything is function of mass, logage, feh.

    Main functionality is interpolation functions that return M, R, mags, etc.
    for given values of mass, age, feh.  
    """
    def __init__(self,m_ini,age,feh,m_act,logL,Teff,logg,mags,tri=None):
        """Warning: if tri object not provided, this will be very slow to be created.
        """

        self.minage = age.min()
        self.maxage = age.max()
        self.minmass = m_act.min()
        self.maxmass = m_act.max()
        self.minfeh = feh.min()
        self.maxfeh = feh.max()
        

        L = 10**logL

        if tri is None:
            points = np.zeros((len(m_ini),2))
            points[:,0] = m_ini
            points[:,1] = age
            self.M = interpnd(points,m_act)
            self.tri = self.M.tri
        else:
            self.tri = tri
            self.M = interpnd(self.tri,m_act)

        self.logL = interpnd(self.tri,logL)
        self.logg = interpnd(self.tri,logg)
        self.logTeff = interpnd(self.tri,np.log10(Teff))

        def Teff_fn(*pts):
            return 10**self.logTeff(*pts)

        self.Teff = Teff_fn
        def R_fn(*pts):
            return np.sqrt(G*self.M(*pts)*MSUN/10**self.logg(*pts))/RSUN
        self.R = R_fn

        self.bands = []
        for band in mags.keys():
            self.bands.append(band)

        self.mag = {band:interpnd(self.tri,mags[band]) for band in self.bands}


    def __call__(self,*args):
        m,age,feh = args 
        Ms = self.M(*args)
        Rs = self.R(*args)
        logLs = self.logL(*args)
        loggs = self.logg(*args)
        Teffs = self.Teff(*args)
        mags = {band:self.mag[band](*args) for band in self.bands}
        
        return {'age':age,'M':Ms,'R':Rs,'logL':logLs,
                'logg':loggs,'Teff':Teffs,'mag':mags}        

    
    def evtrack(self,m,feh=0.0,minage=6.7,maxage=10.17,dage=0.02):
        ages = np.arange(minage,maxage,dage)
        Ms = self.M(m,ages,feh)
        Rs = self.R(m,ages,feh)
        logLs = self.logL(m,ages,feh)
        loggs = self.logg(m,ages,feh)
        Teffs = self.Teff(m,ages,feh)
        mags = {band:self.mag[band](m,ages,feh) for band in self.bands}
        #for band in self.bands:
        #    mags[band] = self.mag[band](m,ages)

        #return array([ages,Ms,Rs,logLs,loggs,Teffs,   #record array?
        return {'age':ages,'M':Ms,'R':Rs,'logL':logLs,'Teff':Teffs,'mag':mags}
            
    def isochrone(self,age,feh=0.0,minm=0.1,maxm=2,dm=0.02):
        ms = np.arange(minm,maxm,dm)
        ages = np.ones(ms.shape)*age

        Ms = self.M(ms,ages,feh)
        Rs = self.R(ms,ages,feh)
        logLs = self.logL(ms,ages,feh)
        loggs = self.logg(ms,ages,feh)
        Teffs = self.Teff(ms,ages,feh)
        mags = {band:self.mag[band](ms,ages,feh) for band in self.bands}
        #for band in self.bands:
        #    mags[band] = self.mag[band](ms,ages)

        return {'M':Ms,'R':Rs,'logL':logLs,'Teff':Teffs,'mag':mags}        
        
    def lhood_fn(self,**kwargs):
        def chisqfn(pars):
            tot = 0
            for kw in kwargs:
                val,err = kwargs[kw]
            if kw in self.bands:
                fn = self.mag[kw]
            else:
                fn = getattr(self,kw)
            tot += (val-fn(*pars))**2/err**2
            return tot
        return chisqfn

    def isofit(self,p0=None,**kwargs):
        """Finds best leastsq match to provided (val,err) keyword pairs.
        
        e.g. isofit(iso,Teff=(5750,50),logg=(4.5,0.1))
        """
        chisqfn = self.lhood_fn(**kwargs)

        if p0 is None:
            p0 = ((self.minmass+self.maxmass)/2,(self.minage + self.maxage)/2.,
                  (self.minfeh + self.maxfeh)/2.)
        pfit = scipy.optimize.fmin(chisqfn,p0,disp=False)
        return self(*pfit)


class StarModel(object):
    def __init__(self,ic,maxAV=1,**kwargs):
        self.ic = ic
        self.properties = kwargs
        self.maxAV = maxAV
        
    def loglike(self,p):
        #add optional distance,reddening params
        mass,age,feh,dist,AV = p
        if mass < self.ic.minmass or mass > self.ic.maxmass \
           or age < self.ic.minage or age > self.ic.maxage \
           or feh < self.ic.minfeh or feh > self.ic.maxfeh:
            return -np.inf
        if dist < 0 or AV < 0:
            return -np.inf
        if AV > self.maxAV:
            return -np.inf

        logl = 0
        for prop in self.properties.keys():
            val,err = self.properties[prop]
            if prop in self.ic.bands:
                mod = self.ic.mag[prop](mass,age,feh) + 5*np.log10(dist) - 5
                A = AV*EXTINCTION[prop]
                mod += A
            elif prop=='feh':
                mod = feh
            else:
                mod = getattr(self.ic,prop)(mass,age,feh)
            logl += -(val-mod)**2/err**2

        if np.isnan(logl):
            logl = -np.inf
        return logl
            
    def fit_mcmc(self,p0=None,nwalkers=200,nburn=200,niter=1000,threads=1):
        if p0 is None:
            p0 = [1,9.3,0.,100,0.4]
        m0 = p0[0]*(1+rand.normal(size=nwalkers)*0.1)
        age0 = p0[1]*(1+rand.normal(size=nwalkers)*0.2)
        feh0 = p0[2] + rand.normal(size=nwalkers)*0.1
        d0 = p0[3]*(1+rand.normal(size=nwalkers)*0.5)
        AV0 = p0[4] + rand.normal(size=nwalkers)*0.2

        p0 = np.array([m0,age0,feh0,d0,AV0]).T

        sampler = emcee.EnsembleSampler(nwalkers,5,self.loglike,threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self.sampler = sampler


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
