from __future__ import division,print_function
import numpy as np
import os,sys,re,os.path
__author__ = 'Timothy D. Morton <tim.morton@gmail.com>'
"""


"""

from scipy.interpolate import LinearNDInterpolator as interpnd
import scipy.optimize
import numpy.random as rand
import emcee

from astropy import constants as const

#Define useful constants
G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

#Read data defining extinction in different bands (relative to A_V)
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



class Isochrone(object):
    """Generic isochrone class. Everything is function of mass, logage, feh.

    Can be instantiated directly, but will typically be used with a pre-defined
    subclass.  

    The following properties are defined as 3-d interpolation functions, all
    taking as arguments (mass,age,feh):

    M, logL, logg, logTeff, Teff, R

    Also defined is a dictionary property 'mag' where self.mag[band] is also
    a similarly-constructed interpolation function.
    
    Parameters
    ----------
    m_ini : array-like
        Initial mass [msun]

    age : array-like
        log_10(age) [yr]

    feh : array-like
        Metallicity

    m_act : array-like
        Actual mass; same as m_ini if mass loss not implemented [msun]

    logL : array-like
        log_10(luminosity) [solar units]

    Teff : array-like
        Effective temperature [K]

    logg : array-like
        log_10(surface gravity) [cgs]

    mags : dict
        dictionary of magnitudes in different bands

    tri : `scipy.spatial.qhull.Delaunay` object, optional
        This is used to initialize the interpolation functions.
        If pre-computed triangulation not provided, then the constructor
        will calculate one.  This might take several minutes, so be patient.
        Much better to use pre-computed ones.
        
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
        """returns properties (or arrays of properties) at given mass, age, feh

        Parameters
        ----------
        mass, age, feh : float or array-like

        Returns
        -------
        values : dictionary
            Dictionary of floats or arrays, containing 'age', 'M',
            'R', 'logL', 'logg', 'Teff', 'mag', where 'mag' is itself
            a dictionary of magnitudes. 
        """
        m,age,feh = args 
        Ms = self.M(*args)
        Rs = self.R(*args)
        logLs = self.logL(*args)
        loggs = self.logg(*args)
        Teffs = self.Teff(*args)
        mags = {band:self.mag[band](*args) for band in self.bands}
        
        return {'age':age,'M':Ms,'R':Rs,'logL':logLs,
                'logg':loggs,'Teff':Teffs,'mag':mags}        

    
    def evtrack(self,m,feh=0.0,minage=None,maxage=None,dage=0.02):
        """Returns evolution track for a single initial mass and feh

        Parameters
        ----------
        m : float
            initial mass of desired track

        feh : float, optional
            metallicity of desired track.  Default = 0.0 (solar)

        minage, maxage : float, optional
            Minimum and maximum log(age) of desired track. Will default
            to min and max age of model isochrones. 

        dage : float, optional
            Spacing in log(age) at which to evaluate models.  Default = 0.02

        Returns
        -------
        values : dictionary
            Dictionary of arrays representing evolution track, containing 'age',
            'M', 'R', 'logL', 'logg', 'Teff', 'mag', where 'mag' is itself
            a dictionary of magnitudes.
        """
        if minage is None:
            minage = self.minage
        if maxage is None:
            maxage = self.maxage
        ages = np.arange(minage,maxage,dage)
        Ms = self.M(m,ages,feh)
        Rs = self.R(m,ages,feh)
        logLs = self.logL(m,ages,feh)
        loggs = self.logg(m,ages,feh)
        Teffs = self.Teff(m,ages,feh)
        mags = {band:self.mag[band](m,ages,feh) for band in self.bands}

        return {'age':ages,'M':Ms,'R':Rs,'logL':logLs,
                'logg':loggs, 'Teff':Teffs, 'mag':mags}
            
    def isochrone(self,age,feh=0.0,minm=None,maxm=None,dm=0.02):
        """Returns stellar models evaluated at a constant age and feh, for a range of masses

        Parameters
        ----------
        age : float
            log(age) of desired isochrone.

        feh : float
            Metallicity of desired isochrone (default = 0.0)

        minm, maxm : float
            Mass range of desired isochrone (will default to max and min available)

        dm : float
            Spacing in mass of desired isochrone

        Returns
        -------
        values : dictionary
            Dictionary of arrays representing evolution track, containing
            'M', 'R', 'logL', 'logg', 'Teff', 'mag', where 'mag' is itself
            a dictionary of magnitudes.

        """
        if minm is None:
            minm = self.minmass
        if maxm is None:
            maxm = self.maxmass
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

        return {'M':Ms,'R':Rs,'logL':logLs,'logg':loggs,
                'Teff':Teffs,'mag':mags}        
        
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

    def maxlike(self,nseeds=10):
        m0 = rand.uniform(self.ic.minmass,self.ic.maxmass,size=nseeds)
        age0 = rand.uniform(8,10,size=nseeds)
        feh0 = rand.uniform(self.ic.minfeh,self.ic.maxfeh,size=nseeds)
        d0 = np.sqrt(rand.uniform(1,1e4**2,size=nseeds))
        AV0 = rand.uniform(0,self.maxAV,size=nseeds)

        costs = np.zeros(nseeds)
        pfits = np.zeros((nseeds,5))
        def fn(p): #fmin is a function *minimizer*
            return -1*self.loglike(p) 
        for i,m,age,feh,d,AV in zip(range(nseeds),
                                    m0,age0,feh0,d0,AV0):
                pfit = scipy.optimize.fmin(fn,[m,age,feh,d,AV],disp=False)
                pfits[i,:] = pfit
                costs[i] = self.loglike(pfit)

        return pfits[np.argmax(costs),:]

            
    def fit_mcmc(self,p0=None,nwalkers=200,nburn=100,niter=500,threads=1):
        if p0 is None:
            m0 = rand.uniform(self.ic.minmass,self.ic.maxmass,size=nwalkers)
            age0 = rand.uniform(8,10,size=nwalkers)
            feh0 = rand.uniform(self.ic.minfeh,self.ic.maxfeh,size=nwalkers)
            
            d0 = np.sqrt(rand.uniform(1,1e4**2,size=nwalkers))
            AV0 = rand.uniform(0,self.maxAV,size=nwalkers)
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
