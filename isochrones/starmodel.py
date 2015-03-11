from __future__ import print_function, division
import os,os.path
import numpy as np
import pandas as pd
import numpy.random as rand
import emcee
import scipy.optimize

from plotutils.plotutils import setfig
import matplotlib.pyplot as plt

try:
    import triangle
except ImportError:
    triangle = None


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

    :param ic: 
        :class:`Isochrone` object used to model star.

    :param maxAV: (optional)
        Maximum allowed extinction (i.e. the extinction @ infinity in direction of star).  Default is 1.

    :param max_distance: (optional)
        Maximum allowed distance (pc).  Default is 1000.
    
    :param **kwargs:
        Keyword arguments must be properties of given isochrone, e.g., logg,
        feh, Teff, and/or magnitudes.  The values represent measurements of
        the star, and must be in (value,error) format. All such keyword
        arguments will be held in ``self.properties``.
        
    """
    def __init__(self,ic,maxAV=1,max_distance=1000,**kwargs):
        self.ic = ic
        self.properties = kwargs
        self.max_distance = max_distance
        self.maxAV = maxAV

    def add_props(self,**kwargs):
        """
        Adds observable properties to ``self.properties``.
        
        """
        for kw,val in kwargs.iteritems():
            self.properties[kw] = val

    def remove_props(self,*args):
        """
        Removes desired properties from ``self.properties``.
        
        """
        for arg in args:
            if arg in self.properties:
                del self.properties[arg]
    
    @property
    def fit_for_distance(self):
        """
        Returns ``True`` if any of the properties are apparent magnitudes.
        
        """
        for prop in self.properties.keys():
            if prop in self.ic.bands:
                return True
        return False
            
    
    def loglike(self,p):
        """Log-likelihood of model at given parameters

        
        :param p : 
            mass, log10(age), feh, [distance, A_V (extinction)].
            Final two should only be provided if ``self.fit_for_distance``
            is ``True``; that is, apparent magnitudes are provided.
            

        Returns log-likelihood.  Will be -np.inf if values out of range.
        
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
        if self.fit_for_distance:
            if dist < 0 or AV < 0 or dist > self.max_distance:
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

        #IMF prior
        logl += np.log(salpeter_prior(mass))
        
        #distance prior?
        

        ##prior to sample ages with linear prior
        #a0 = 10**self.ic.minage
        #a1 = 10**self.ic.maxage
        #da = a1-a0
        #a = 10**age
        #logl += np.log(a/(a1-a0))

        return logl

    def maxlike(self,nseeds=50):
        """Returns the best-fit parameters, choosing the best of multiple starting guesses

        :param nseeds: (optional)
            Number of starting guesses, uniformly distributed throughout
            allowed ranges.  Default=50.

        Returns list of best-fit parameters: [m,age,feh,[distance,A_V]].
        Note that distance and A_V values will be meaningless unless
        magnitudes are present in ``self.properties``.
        
        """
        m0,age0,feh0 = self.ic.random_points(nseeds)
        d0 = 10**(rand.uniform(0,np.log10(self.max_distance),size=nseeds))
        AV0 = rand.uniform(0,self.maxAV,size=nseeds)

        

        costs = np.zeros(nseeds)

        if self.fit_for_distance:
            pfits = np.zeros((nseeds,5))
        else:
            pfits = np.zeros((nseeds,3))
            
        def fn(p): #fmin is a function *minimizer*
            return -1*self.loglike(p)
        
        for i,m,age,feh,d,AV in zip(range(nseeds),
                                    m0,age0,feh0,d0,AV0):
                if self.fit_for_distance:
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
        sampler 
        """

        if self.fit_for_distance:
            npars = 5
            if initial_burn is None:
                initial_burn = True
        else:
            if initial_burn is None:
                initial_burn = False
            npars = 3

        if p0 is None:
            m0,age0,feh0 = self.ic.random_points(nwalkers)
            #d0 = np.sqrt(rand.uniform(1,self.max_distance**2,size=nwalkers))
            d0 = 10**(rand.uniform(0,np.log10(self.max_distance),size=nwalkers))
            AV0 = rand.uniform(0,self.maxAV,size=nwalkers)
            if self.fit_for_distance:
                p0 = np.array([m0,age0,feh0,d0,AV0]).T
            else:
                p0 = np.array([m0,age0,feh0]).T
            if initial_burn:
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike,
                                                threads=threads)
                #ninitial = 300 #should this be parameter?
                pos, prob, state = sampler.run_mcmc(p0, ninitial) 
                wokinds = np.where((sampler.naccepted/ninitial > 0.15) &
                                   (sampler.naccepted/ninitial < 0.4))[0]
                i=1
                while len(wokinds)==0:
                    thresh = 0.15 - i*0.02
                    if thresh < 0:
                        raise RuntimeError('Initial burn has no acceptance?')
                    wokinds = np.where((sampler.naccepted/ninitial > thresh) &
                                       (sampler.naccepted/ninitial < 0.4))[0]
                    i += 1
                inds = rand.randint(len(wokinds),size=nwalkers)
                p0 = sampler.chain[wokinds[inds],:,:].mean(axis=1) #reset p0
                p0 *= (1 + rand.normal(size=p0.shape)*0.01)
        else:
            p0 = np.array(p0)
            p0 = rand.normal(size=(nwalkers,npars))*0.01 + p0.T[None,:]
            if self.fit_for_distance:
                p0[:,3] *= (1 + rand.normal(size=nwalkers)*0.5)
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike,threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self._sampler = sampler
        return sampler

    def triangle_plots(self, basename=None, **kwargs):
        fig1 = self.triangle(plot_datapoints=False,
                            params=['mass','radius','Teff','feh','age'])
        if basename is not None:
            plt.savefig('{}_physical.png'.format(basename))
            plt.close()
        fig2 = self.prop_triangle()
        if basename is not None:
            plt.savefig('{}_observed.png'.format(basename))
            plt.close()
        return fig1, fig2

    def triangle(self, params=None, query=None, extent=0.99,
                 **kwargs):
        if triangle is None:
            raise ImportError('please run "pip install triangle_plot".')
        
        if params is None:
            if self.fit_for_distance:
                params = ['mass', 'age', 'feh', 'distance', 'AV']
            else:
                params = ['mass', 'age', 'feh']

        df = self.samples
        if query is not None:
            df = df.query(query)

        extents = [extent for foo in params]

        return triangle.corner(df[params], labels=params, 
                               extents=extents, **kwargs)


    def prop_triangle(self, **kwargs):
        truths = []
        params = []
        for p in self.properties:
            if p in self.ic.bands:
                params.append('{}_mag'.format(p))
            else:
                params.append(p)
            truths.append(self.properties[p][0])
        return self.triangle(params, truths=truths, **kwargs)
        

    @property
    def sampler(self):
        if hasattr(self,'_sampler'):
            return self._sampler
        else:
            raise AttributeError('MCMC must be run to access sampler')

    @property
    def samples(self):
        """Dataframe with samples drawn from isochrone according to posterior

        Culls samples to have lnlike within 10 of max lnlike (hard-coded)
        """
        if not hasattr(self,'sampler') and not hasattr(self, '_samples'):
            raise AttributeError('Must run MCMC (or load from file) before accessing samples')
        
        try:
            df = self._samples.copy()

        except AttributeError:
            max_lnlike = self.sampler.flatlnprobability.max()
            ok = self.sampler.flatlnprobability > (max_lnlike - 10)
            
            mass = self.sampler.flatchain[:,0][ok]
            age = self.sampler.flatchain[:,1][ok]
            feh = self.sampler.flatchain[:,2][ok]
            
            if self.fit_for_distance:
                distance = self.sampler.flatchain[:,3][ok]
                AV = self.sampler.flatchain[:,4][ok]
            
            df = self.ic(mass, age, feh)
            df['age'] = age
            df['feh'] = feh
            
            if self.fit_for_distance:
                df['distance'] = distance
                df['AV'] = AV
                
            self._samples = df.copy()

        if self.fit_for_distance:
            dm = 5*np.log10(df['distance']) - 5
            for b in self.ic.bands:
                df['{}_mag'.format(b)] += dm

        return df

    def random_samples(self, n):
        samples = self.samples
        inds = rand.randint(len(samples),size=n)

        newsamples = samples.iloc[inds]
        newsamples.reset_index(inplace=True)
        return newsamples


    def prop_samples(self,prop,return_values=True,conf=0.683):
        """Returns samples of given property, based on MCMC sampling

        Parameters
        ----------
        prop : str
            Desired property. Options are any valid property of `self.ic`.
            If MCMC hasn't been run, a call to this function will run MCMC.
            'distance' and 'AV' will only work if magnitudes are provided
            as properties.

        Returns
        -------
        chain : array
            Posterior sampling of given property.
        """
        samples = self.samples[prop]
        
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

    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """Saves object data to HDF file (only works if MCMC is run)
        """
        
        if os.path.exists(filename):
            store = pd.HDFStore(filename)
            if path in store:
                store.close()
                if overwrite:
                    os.remove(filename)
                elif not append:
                    raise IOError('{} in {} exists.  Set either overwrite or append option.'.format(path,filename))
            else:
                store.close()

        self.samples.to_hdf(filename, '{}/samples'.format(path))

        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/samples'.format(path)).attrs
        attrs.properties = self.properties
        attrs.ic_type = type(self.ic)
        attrs.maxAV = self.maxAV
        attrs.max_distance = self.max_distance
        store.close()

    @classmethod
    def load_hdf(cls, filename, path=''):

        store = pd.HDFStore(filename)
        try:
            samples = store['{}/samples'.format(path)]
            attrs = store.get_storer('{}/samples'.format(path)).attrs        
        except:
            store.close()
            raise
        properties = attrs.properties
        maxAV = attrs.maxAV
        max_distance = attrs.max_distance
        ic_type = attrs.ic_type
        store.close()

        ic = ic_type()
        mod = cls(ic, maxAV=maxAV, max_distance=max_distance,
                  **properties)
        mod._samples = samples
        return mod


def salpeter_prior(m,alpha=-2.35,minmass=0.1,maxmass=10):
    C = (1+alpha)/(maxmass**(1+alpha)-minmass**(1+alpha))
    if m < minmass or m > maxmass:
        return 0
    else:
        return C*m**(alpha)

