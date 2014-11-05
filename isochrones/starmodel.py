import os,os.path
import numpy as np
import numpy.random as rand
import emcee
import scipy.optimize

from plotutils.plotutils import setfig
import matplotlib.pyplot as plt

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

    def add_props(self,**kwargs):
        for kw,val in kwargs.iteritems():
            self.properties[kw] = val

    def remove_props(self,*args):
        for arg in args:
            if arg in self.properties:
                del self.properties[arg]
    
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

        logl += np.log(salpeter_prior(mass)) #IMF prior
        
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
        m0,age0,feh0 = self.ic.random_points(nseeds)
        d0 = np.sqrt(rand.uniform(1,1e6,size=nseeds))
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
                 p0=None,initial_burn=None,
                 ninitial=300):
        """Fits stellar model using MCMC.

        Parameters
        ----------
        nwalkers, nburn, niter, threads : int
            Parameters to pass to emcee sampling for MCMC.

        p0 : array-like
            Initial parameters for emcee.  If not provided, then chains
            will behave according to whether inital_burn is set.

        initial_burn : bool or None, optional
            If `True`, then initialize walkers first with a random initialization,
            then cull the walkers, keeping only those with > 15% acceptance
            rate, then reinitialize sampling.  If `False`, then just do
            normal burn-in.  Default is `None`, which will be set to `True` if
            fitting for distance (i.e., if there are apparent magnitudes as
            properties of the model), and `False` if not.

        ninitial : int
            Number of iterations to test walkers for acceptance rate before
            re-initializing.

        Returns
        -------
        None, but defines self.sampler that holds the results of fit.
        """
        fit_for_distance = self.fit_for_distance()
        if fit_for_distance:
            npars = 5
            if initial_burn is None:
                initial_burn = True
        else:
            if initial_burn is None:
                initial_burn = False
            npars = 3

        if p0 is None:
            m0,age0,feh0 = self.ic.random_points(nwalkers)
            d0 = np.sqrt(rand.uniform(1,1e6,size=nwalkers))
            AV0 = rand.uniform(0,self.maxAV,size=nwalkers)
            if fit_for_distance:
                p0 = np.array([m0,age0,feh0,d0,AV0]).T
            else:
                p0 = np.array([m0,age0,feh0]).T
            if initial_burn:
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike,
                                                threads=threads)
                #ninitial = 300 #should this be parameter?
                pos, prob, state = sampler.run_mcmc(p0, ninitial) 
                wokinds = np.where(sampler.naccepted/ninitial > 0.15)[0]
                inds = rand.randint(len(wokinds),size=nwalkers)
                p0 = sampler.chain[wokinds[inds],:,:].mean(axis=1) #reset p0
        else:
            p0 = np.array(p0)
            p0 = rand.normal(size=(nwalkers,npars))*0.01 + p0.T[None,:]
            if fit_for_distance:
                p0[:,3] *= (1 + rand.normal(size=nwalkers)*0.5)
        
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

def salpeter_prior(m,alpha=-2.35,minmass=0.1,maxmass=10):
    C = (1+alpha)/(maxmass**(1+alpha)-minmass**(1+alpha))
    if m < minmass or m > maxmass:
        return 0
    else:
        return C*m**(alpha)
