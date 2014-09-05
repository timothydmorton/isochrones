from __future__ import division,print_function

__author__ = 'Timothy D. Morton <tim.morton@gmail.com>'
"""


"""

import numpy as np
import os,sys,re,os.path

from scipy.interpolate import LinearNDInterpolator as interpnd
import scipy.optimize
import numpy.random as rand
import emcee

from astropy import constants as const

import matplotlib.pyplot as plt
from plotutils.plotutils import setfig

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

    The following methods are implemented as 3-d interpolation functions, all
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
            self.mass = interpnd(points,m_act)
            self.tri = self.mass.tri
        else:
            self.tri = tri
            self.mass = interpnd(self.tri,m_act)

        self.logL = interpnd(self.tri,logL)
        self.logg = interpnd(self.tri,logg)
        self.logTeff = interpnd(self.tri,np.log10(Teff))

        def Teff_fn(*pts):
            return 10**self.logTeff(*pts)
        self.Teff = Teff_fn
        
        def R_fn(*pts):
            return np.sqrt(G*self.mass(*pts)*MSUN/10**self.logg(*pts))/RSUN
        self.radius = R_fn

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
        Ms = self.mass(*args)
        Rs = self.radius(*args)
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
        Ms = self.mass(m,ages,feh)
        Rs = self.radius(m,ages,feh)
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

        Ms = self.mass(ms,ages,feh)
        Rs = self.radius(ms,ages,feh)
        logLs = self.logL(ms,ages,feh)
        loggs = self.logg(ms,ages,feh)
        Teffs = self.Teff(ms,ages,feh)
        mags = {band:self.mag[band](ms,ages,feh) for band in self.bands}
        #for band in self.bands:
        #    mags[band] = self.mag[band](ms,ages)

        return {'M':Ms,'R':Rs,'logL':logLs,'logg':loggs,
                'Teff':Teffs,'mag':mags}        
        


class StarModel(object):
    """An object to represent a star, with observed properties, modeled by an Isochrone

    Parameters
    ----------
    ic : `Isochrone` object
        Isochrone object used to model star.

    maxAV : float
        Maximum allowed extinction (i.e. the extinction @ infinity in direction of star)

    kwargs
        Keyword arguments must be properties of given isochrone, e.g.,
        logg, feh, Teff, and/or magnitudes.  The values represent measurements of
        the star, and must be in (value,error) format.
    """
    def __init__(self,ic,maxAV=1,**kwargs):
        self.ic = ic
        self.properties = kwargs

        self.maxAV = maxAV

    def fit_for_distance(self):
        for prop in self.properties.keys():
            if prop in self.ic.bands:
                return True
        return False
            
    
    def loglike(self,p):
        """Log-likelihood of model at given parameters

        Parameters
        ----------
        p : [float,float,float,float,float] or [float,float,float]
            mass, log(age), feh, [distance, A_V (extinction)]
            

        Returns
        -------
        logl : float
            log-likelihood.  Will be -np.inf if values out of range.
        """
        if len(p)==5:
            fit_for_distance = True
            mass,age,feh,dist,AV = p
        elif len(p)==3:
            fit_for_distance = False
            mass,age,feh = p
                        
        if mass < self.ic.minmass or mass > self.ic.maxmass \
           or age < self.ic.minage or age > self.ic.maxage \
           or feh < self.ic.minfeh or feh > self.ic.maxfeh:
            return -np.inf
        if fit_for_distance:
            if dist < 0 or AV < 0:
                return -np.inf
            if AV > self.maxAV:
                return -np.inf

        logl = 0
        for prop in self.properties.keys():
            val,err = self.properties[prop]
            if prop in self.ic.bands:
                if not fit_for_distance:
                    raise ValueError('must fit for mass, age, feh, dist, A_V if apparent magnitudes provided.')
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

        #print('{:.2f} {:.2f} {:.2f}: {:.4g}'.format(mass,age,feh,logl))
        return logl

    def maxlike(self,nseeds=50):
        """Returns the best-fit parameters, choosing the best of multiple starting guesses

        Parameters
        ----------
        nseeds : int
            Number of starting guesses, uniformly distributed throughout
            allowed ranges.

        Returns
        -------
        pfit : list
            [m,age,feh,[distance,A_V]] best-fit parameters.  Note that distance
            and A_V values will be meaningless unless magnitudes are provided.
        """
        m0 = rand.uniform(self.ic.minmass,self.ic.maxmass,size=nseeds)
        age0 = rand.uniform(8,10,size=nseeds)
        feh0 = rand.uniform(self.ic.minfeh,self.ic.maxfeh,size=nseeds)
        d0 = np.sqrt(rand.uniform(1,1e4**2,size=nseeds))
        AV0 = rand.uniform(0,self.maxAV,size=nseeds)

        costs = np.zeros(nseeds)
        fit_for_distance = self.fit_for_distance()

        if fit_for_distance:
            pfits = np.zeros((nseeds,5))
        else:
            pfits = np.zeros((nseeds,3))
            
        def fn(p): #fmin is a function *minimizer*
            return -1*self.loglike(p)
        
        for i,m,age,feh,d,AV in zip(range(nseeds),
                                    m0,age0,feh0,d0,AV0):
                if fit_for_distance:
                    pfit = scipy.optimize.fmin(fn,[m,age,feh,d,AV],disp=False)
                else:
                    pfit = scipy.optimize.fmin(fn,[m,age,feh],disp=False)
                pfits[i,:] = pfit
                costs[i] = self.loglike(pfit)

        return pfits[np.argmax(costs),:]

            
    def fit_mcmc(self,nwalkers=200,nburn=100,niter=200,threads=1,
                 p0=None,maxlike_nseeds=50):
        """Fits stellar model using MCMC.

        Parameters
        ----------
        nwalkers, nburn, niter, threads : int
            Parameters to pass to emcee sampling for MCMC.

        p0 : array-like
            Initial parameters for emcee.  If not provided, `self.maxlike`
            will be called to initialize parameter ball.

        maxlike_nseeds : int
            Number of seeds for `maxlike` call.  Default=50.

        Returns
        -------
        None, but defines self.sampler that holds the results of fit.
        """
        fit_for_distance = self.fit_for_distance()
        if fit_for_distance:
            npars = 5
        else:
            npars = 3

        if p0 is None:
            # use ball around maxlike params to initialize walkers
            p0 = self.maxlike(maxlike_nseeds)
        p0 = rand.normal(size=(nwalkers,npars))*0.1 + p0.T[None,:]
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike,threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self.sampler = sampler

    def prop_samples(self,prop,return_values=True,conf=0.683):
        """Returns samples of given property, based on MCMC sampling

        Parameters
        ----------
        prop : str
            Desired property. Options are 'mass', 'radius', 'age',
            'logg', 'logL', 'Teff', 'feh', 'distance', 'AV',
            or any valid passband name for `self.ic` (`Isochrone` object).
            If MCMC hasn't been run, a call to this function will run MCMC.
            'distance' and 'AV' will only work if magnitudes are provided
            as properties.

        Returns
        -------
        chain : array
            Posterior sampling of given property.
        """
        if not hasattr(self,'sampler'):
            self.fit_mcmc()

        if prop=='age':
            samples = self.sampler.flatchain[:,1]
        elif prop=='feh':
            samples = self.sampler.flatchain[:,2]
        elif prop=='distance':
            samples = self.sampler.flatchain[:,3]
        elif prop=='AV':
            samples = self.sampler.flatchain[:,4]
        else:           
            if prop in self.ic.bands:
                fn = self.ic.mag[prop]
            else:
                fn = getattr(self.ic,prop)

            samples = fn(self.sampler.flatchain[:,:3]) #excluding dist,A_V if present
        
        if return_values:
            sorted = np.sort(samples)
            med = np.median(samples)
            n = len(samples)
            lo_ind = int(n*(0.5 - conf/2))
            hi_ind = int(n*(0.5 + conf/2))
            lo = med - sorted[lo_ind]
            hi = sorted[hi_ind] - med
            return samples, (med,lo,hi)
        else:
            return samples 

    def plot_samples(self,prop,fig=None,label=True,
                     histtype='step',bins=50,lw=3,
                     **kwargs):
        """Plots histogram of samples of desired property.

        Parameters
        ----------
        prop : str
           Desired property.  See `prop_samples` for appropriate names.

        fig : None, or int
           Argument for `plotutils.setfig`.

        histtype,bins,lw : various
            Arguments passed to `plt.hist`

        kwargs
            Keyword arguments passed to `plt.hist`
        """
        setfig(fig)
        samples,stats = self.prop_samples(prop)
        plt.hist(samples,bins=bins,normed=True,
                 histtype=histtype,lw=lw,**kwargs)
        plt.xlabel(prop)
        plt.ylabel('Normalized count')
        
        if label:
            med,lo,hi = stats
            plt.annotate('$%.2f^{+%.2f}_{-%.2f}$' % (med,hi,lo),
                         xy=(0.7,0.8),xycoords='axes fraction',fontsize=20)
