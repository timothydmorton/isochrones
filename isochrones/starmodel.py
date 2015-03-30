from __future__ import print_function, division
import os,os.path
import numpy as np
import logging
import re

try:
    import pandas as pd
except ImportError:
    pd = None
    
import numpy.random as rand
import scipy.optimize

try:
    import emcee
except ImportError:
    emcee = None

try:
    from plotutils.plotutils import setfig
except ImportError:
    setfig = None
    
import matplotlib.pyplot as plt

try:
    import triangle
except ImportError:
    triangle = None


from .extinction import EXTINCTION

class StarModel(object):
    """An object to represent a star, with observed properties, modeled by an Isochrone

    This is used to fit a physical stellar model to observed
    quantities, e.g. spectroscopic or photometric, based on
    an :class:`Isochrone`.

    Note that by default a local metallicity prior, based on SDSS data,
    will be used when :func:`StarModel.fit_mcmc` is called.

    :param ic: 
        :class:`Isochrone` object used to model star.

    :param maxAV: (optional)
        Maximum allowed extinction (i.e. the extinction @ infinity in direction of star).  Default is 1.

    :param max_distance: (optional)
        Maximum allowed distance (pc).  Default is 3000.
    
    :param **kwargs:
        Keyword arguments must be properties of given isochrone, e.g., logg,
        feh, Teff, and/or magnitudes.  The values represent measurements of
        the star, and must be in (value,error) format. All such keyword
        arguments will be held in ``self.properties``.
        
    """
    def __init__(self,ic,maxAV=1,max_distance=3000,**kwargs):
        self.ic = ic
        self.properties = kwargs
        self.max_distance = max_distance
        self.maxAV = maxAV
        self._samples = None

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
        ``True`` if any of the properties are apparent magnitudes.
        
        """
        for prop in self.properties.keys():
            if prop in self.ic.bands:
                return True
        return False
            
    
    def loglike(self,p, use_local_fehprior=True):
        """Log-likelihood of model at given parameters

        
        :param p: 
            mass, log10(age), feh, [distance, A_V (extinction)].
            Final two should only be provided if ``self.fit_for_distance``
            is ``True``; that is, apparent magnitudes are provided.
            
        :param use_local_fehprior:
            Whether to use the Casagrande et al. (2011) prior via
            :func:`localfehdist`.  Default is ``True``.

        :return:
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
        
        #distance prior ~d^2 out to d_max
        if fit_for_distance:
            logl += np.log(3/self.max_distance**3 * dist**2)

        if use_local_fehprior:
            #From Jo Bovy:
            #https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
            #2D gaussian fit based on Casagrande (2011)

            fehdist= 0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
                +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.)
            logl += np.log(fehdist)

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

        :return:
            list of best-fit parameters: ``[m,age,feh,[distance,A_V]]``.
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

            
    def fit_mcmc(self,nwalkers=200,nburn=100,niter=200,
                 p0=None,initial_burn=None,
                 ninitial=100, loglike_kwargs=None,
                 **kwargs):
        """Fits stellar model using MCMC.

        :param nwalkers: (optional)
            Number of walkers to pass to :class:`emcee.EnsembleSampler`.
            Default is 200.

        :param nburn: (optional)
            Number of iterations for "burn-in."  Default is 100.

        :param niter: (optional)
            Number of for-keeps iterations for MCMC chain.
            Default is 200.

        :param p0: (optional)
            Initial parameters for emcee.  If not provided, then chains
            will behave according to whether inital_burn is set.

        :param initial_burn: (optional)
            If `True`, then initialize walkers first with a random initialization,
            then cull the walkers, keeping only those with > 15% acceptance
            rate, then reinitialize sampling.  If `False`, then just do
            normal burn-in.  Default is `None`, which will be set to `True` if
            fitting for distance (i.e., if there are apparent magnitudes as
            properties of the model), and `False` if not.
            
        :param ninitial: (optional)
            Number of iterations to test walkers for acceptance rate before
            re-initializing.

        :param loglike_args:
            Any arguments to pass to :func:`StarModel.loglike`, such 
            as what priors to use.
        
        :return:
            :class:`emcee.EnsembleSampler` object.
            
        """

        #clear any saved _samples
        if self._samples is not None:
            self._samples = None
            

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
            d0 = 10**(rand.uniform(0,np.log10(self.max_distance),size=nwalkers))
            AV0 = rand.uniform(0,self.maxAV,size=nwalkers)
            if self.fit_for_distance:
                p0 = np.array([m0,age0,feh0,d0,AV0]).T
            else:
                p0 = np.array([m0,age0,feh0]).T
            if initial_burn:
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike,
                                                **kwargs)
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
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self._sampler = sampler
        return sampler

    def triangle_plots(self, basename=None, format='png',
                       **kwargs):
        """Returns two triangle plots, one with physical params, one observational

        :param basename:
            If basename is provided, then plots will be saved as
            "[basename]_physical.[format]" and "[basename]_observed.[format]"

        :param format:
            Format in which to save figures (e.g., 'png' or 'pdf')

        :param **kwargs:
            Additional keyword arguments passed to :func:`StarModel.triangle`
            and :func:`StarModel.prop_triangle`

        :return:
             * Physical parameters triangle plot (mass, radius, Teff, feh, age, distance)
             * Observed properties triangle plot.
             
        """
        if self.fit_for_distance:
            fig1 = self.triangle(plot_datapoints=False,
                                 params=['mass','radius','Teff','feh','age','distance'],
                                 **kwargs)
        else:
            fig1 = self.triangle(plot_datapoints=False,
                                 params=['mass','radius','Teff','feh','age'],
                                 **kwargs)
            

        if basename is not None:
            plt.savefig('{}_physical.{}'.format(basename,format))
            plt.close()
        fig2 = self.prop_triangle(**kwargs)
        if basename is not None:
            plt.savefig('{}_observed.{}'.format(basename,format))
            plt.close()
        return fig1, fig2

    def triangle(self, params=None, query=None, extent=0.999,
                 **kwargs):
        """
        Makes a nifty corner plot.

        Uses :func:`triangle.corner`.

        :param params: (optional)
            Names of columns (from :attr:`StarModel.samples`)
            to plot.  If ``None``, then it will plot samples
            of the parameters used in the MCMC fit-- that is,
            mass, age, [Fe/H], and optionally distance and A_V.

        :param query: (optional)
            Optional query on samples.

        :param extent: (optional)
            Will be appropriately passed to :func:`triangle.corner`.

        :param **kwargs:
            Additional keyword arguments passed to :func:`triangle.corner`.

        :return:
            Figure oject containing corner plot.
            
        """
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

        #convert extent to ranges, but making sure
        # that truths are in range.
        extents = []
        for i,par in enumerate(params):
            qs = np.array([0.5 - 0.5*extent, 0.5 + 0.5*extent])
            minval, maxval = self.samples[par].quantile(qs)
            if 'truths' in kwargs:
                datarange = maxval - minval
                if kwargs['truths'][i] < minval:
                    minval = kwargs['truths'][i] - 0.05*datarange
                if kwargs['truths'][i] > maxval:
                    maxval = kwargs['truths'][i] + 0.05*datarange
            extents.append((minval,maxval))
            

        return triangle.corner(df[params], labels=params, 
                               extents=extents, **kwargs)


    def prop_triangle(self, **kwargs):
        """
        Makes corner plot of only observable properties.

        The idea here is to compare the predictions of the samples
        with the actual observed data---this can be a quick way to check
        if there are outlier properties that aren't predicted well
        by the model.

        :param **kwargs:
            Keyword arguments passed to :func:`StarModel.triangle`.

        :return:
            Figure object containing corner plot.
         
        """
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
        """
        Sampler object from MCMC run.
        """
        if hasattr(self,'_sampler'):
            return self._sampler
        else:
            raise AttributeError('MCMC must be run to access sampler')

    def _make_samples(self, lnprob_thresh=0.005):

        #cull points in lowest 0.5% of lnprob
        lnprob_thresh = np.percentile(self.sampler.flatlnprobability, 
                                      lnprob_thresh*100)
        ok = self.sampler.flatlnprobability > lnprob_thresh
            
        mass = self.sampler.flatchain[:,0][ok]
        age = self.sampler.flatchain[:,1][ok]
        feh = self.sampler.flatchain[:,2][ok]
            
        if self.fit_for_distance:
            distance = self.sampler.flatchain[:,3][ok]
            AV = self.sampler.flatchain[:,4][ok]
        else:
            distance = None
            AV = 0

        df = self.ic(mass, age, feh, 
                     distance=distance, AV=AV)
        df['age'] = age
        df['feh'] = feh
            
        if self.fit_for_distance:
            df['distance'] = distance
            df['AV'] = AV
                
        self._samples = df.copy()
        

    @property
    def samples(self):
        """Dataframe with samples drawn from isochrone according to posterior

        Culls samples to drop lowest 0.5% of lnprob values.

        Columns include both the sampling parameters from the MCMC
        fit (mass, age, Fe/H, [distance, A_V]), and also evaluation
        of the :class:`Isochrone` at each of these sample points---this
        is how chains of physical/observable parameters get produced.
        
        """
        if not hasattr(self,'sampler') and self._samples is None:
            raise AttributeError('Must run MCMC (or load from file) '+
                                 'before accessing samples')
        
        if self._samples is not None:
            df = self._samples
        else:
            self._make_samples()
            df = self._samples

        return df

    def random_samples(self, n):
        """
        Returns a random sampling of given size from the existing samples.

        :param n:
            Number of samples

        :return:
            :class:`pandas.DataFrame` of length ``n`` with random samples.
        """
        samples = self.samples
        inds = rand.randint(len(samples),size=int(n))

        newsamples = samples.iloc[inds]
        newsamples.reset_index(inplace=True)
        return newsamples


    def prop_samples(self,prop,return_values=True,conf=0.683):
        """Returns samples of given property, based on MCMC sampling

        :param prop:
            Name of desired property.  Must be column of ``self.samples``.

        :param return_values: (optional)
            If ``True`` (default), then also return (median, lo_err, hi_err)
            corresponding to desired credible interval.

        :param conf: (optional)
            Desired quantile for credible interval.  Default = 0.683.

        :return:
            :class:`np.ndarray` of desired samples

        :return: 
            Optionally also return summary statistics (median, lo_err, hi_err),
            if ``returns_values == True`` (this is default behavior)
        
        """
        samples = self.samples[prop].values
        
        if return_values:
            sorted = np.sort(samples)
            med = np.median(samples)
            n = len(samples)
            lo_ind = int(n*(0.5 - conf/2))
            hi_ind = int(n*(0.5 + conf/2))
            lo = med - sorted[lo_ind]
            hi = sorted[hi_ind] - med
            return samples.values, (med,lo,hi)
        else:
            return samples.values

    def plot_samples(self,prop,fig=None,label=True,
                     histtype='step',bins=50,lw=3,
                     **kwargs):
        """Plots histogram of samples of desired property.

        :param prop:
            Desired property (must be legit column of samples)
            
        :param fig:
              Argument for :func:`plotutils.setfig` (``None`` or int).

        :param histtype, bins, lw:
             Passed to :func:`plt.hist`.

        :param **kwargs:
            Additional keyword arguments passed to `plt.hist`

        :return:
            Figure object.
        """
        setfig(fig)
        samples,stats = self.prop_samples(prop)
        fig = plt.hist(samples,bins=bins,normed=True,
                 histtype=histtype,lw=lw,**kwargs)
        plt.xlabel(prop)
        plt.ylabel('Normalized count')
        
        if label:
            med,lo,hi = stats
            plt.annotate('$%.2f^{+%.2f}_{-%.2f}$' % (med,hi,lo),
                         xy=(0.7,0.8),xycoords='axes fraction',fontsize=20)

        return fig
            
    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """Saves object data to HDF file (only works if MCMC is run)

        Samples are saved to /samples location under given path,
        and object properties are also attached, so suitable for
        re-loading via :func:`StarModel.load_hdf`.
        
        :param filename:
            Name of file to save to.  Should be .h5 file.

        :param path: (optional)
            Path within HDF file structure to save to.

        :param overwrite: (optional)
            If ``True``, delete any existing file by the same name
            before writing.

        :param append: (optional)
            If ``True``, then if a file exists, then just the path
            within the file will be updated.
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
        """
        A class method to load a saved StarModel from an HDF5 file.

        File must have been created by a call to :func:`StarModel.save_hdf`.

        :param filename:
            H5 file to load.

        :param path: (optional)
            Path within HDF file.

        :return:
            :class:`StarModel` object.
        """
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


class BinaryStarModel(StarModel):
    """
    Object used to fit two stars at the same distance to given observed properties

    Initialize the same way as :class:`StarModel`.

    Difference between this object and a regular :class:`StarModel` is that
    the fit parameters include two masses: ``mass_A`` and ``mass_B`` instead
    of just one.



    """
    def loglike(self, p, use_local_fehprior=True):
        """Log-likelihood of model at given parameters

        
        :param p: 
            mass_A, mass_B, log10(age), feh, [distance, A_V (extinction)].
            Final two should only be provided if ``self.fit_for_distance``
            is ``True``; that is, apparent magnitudes are provided.
            
        :param use_local_fehprior:
            Whether to use the Casagrande et al. (2011) prior via
            :func:`localfehdist`.  Default is ``True``.

        :return:
           log-likelihood.  Will be -np.inf if values out of range.
        
        """
        if len(p)==6:
            fit_for_distance = True
            mass_A, mass_B, age, feh, dist, AV = p
        elif len(p)==4:
            fit_for_distance = False
            mass_A, mass_B, age, feh = p
                        
        #keep values in range; enforce mass_A > mass_B
        if mass_A < self.ic.minmass or mass_A > self.ic.maxmass \
           or mass_B < self.ic.minmass or mass_B > self.ic.maxmass \
           or mass_B > mass_A \
           or age < self.ic.minage or age > self.ic.maxage \
           or feh < self.ic.minfeh or feh > self.ic.maxfeh:
            return -np.inf
        if fit_for_distance:
            if dist < 0 or AV < 0 or dist > self.max_distance:
                return -np.inf
            if AV > self.maxAV:
                return -np.inf

        logl = 0
        for prop in self.properties.keys():
            val,err = self.properties[prop]
            if prop in self.ic.bands:
                if not fit_for_distance:
                    raise ValueError('must fit for mass, age, feh, dist,'+ 
                                     'A_V if apparent magnitudes provided.')
                mods = self.ic.mag[prop]([mass_A, mass_B], 
                                         age, feh) + 5*np.log10(dist) - 5
                A = AV*EXTINCTION[prop]
                mods += A
                mod = addmags(*mods)
            elif prop=='feh':
                mod = feh
            else:
                mod = getattr(self.ic,prop)(mass_A,age,feh)
            logl += -(val-mod)**2/err**2

        if np.isnan(logl):
            logl = -np.inf


        #IMF prior
        logl += np.log(salpeter_prior(mass_A))
        
        #distance prior ~d^2 out to d_max
        if fit_for_distance:
            logl += np.log(3/self.max_distance**3 * dist**2)

        if use_local_fehprior:
            #From Jo Bovy:
            #https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
            #2D gaussian fit based on Casagrande (2011)

            fehdist= 0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
                +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.)
            logl += np.log(fehdist)

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

        :return:
            list of best-fit parameters: ``[mA,mB,age,feh,[distance,A_V]]``.
            Note that distance and A_V values will be meaningless unless
            magnitudes are present in ``self.properties``.
        
        """
        mA_0,age0,feh0 = self.ic.random_points(nseeds)
        mB_0,foo1,foo2 = self.ic.random_points(nseeds)
        mA_fixed = np.maximum(mA_0,mB_0)
        mB_fixed = np.minimum(mA_0,mB_0)
        mA_0, mB_0 = (mA_fixed, mB_fixed)

        d0 = 10**(rand.uniform(0,np.log10(self.max_distance),size=nseeds))
        AV0 = rand.uniform(0,self.maxAV,size=nseeds)

        

        costs = np.zeros(nseeds)

        if self.fit_for_distance:
            pfits = np.zeros((nseeds,6))
        else:
            pfits = np.zeros((nseeds,4))
            
        def fn(p): #fmin is a function *minimizer*
            return -1*self.loglike(p)
        
        for i,mA,mB,age,feh,d,AV in zip(range(nseeds),
                                    mA_0,mB_0,age0,feh0,d0,AV0):
                if self.fit_for_distance:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,age,feh,d,AV],disp=False)
                else:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,age,feh],disp=False)
                pfits[i,:] = pfit
                costs[i] = self.loglike(pfit)

        return pfits[np.argmax(costs),:]

            
    def fit_mcmc(self,nwalkers=200,nburn=100,niter=200,
                 p0=None,initial_burn=None,
                 ninitial=100, loglike_kwargs=None,
                 **kwargs):
        """Fits stellar model using MCMC.

        See :func:`StarModel.fit_mcmc`
        """

        #clear any saved _samples
        if self._samples is not None:
            self._samples = None
            

        if self.fit_for_distance:
            npars = 6
            if initial_burn is None:
                initial_burn = True
        else:
            if initial_burn is None:
                initial_burn = False
            npars = 4

        if p0 is None:
            mA_0,age0,feh0 = self.ic.random_points(nwalkers)
            mB_0,foo1,foo2 = self.ic.random_points(nwalkers)
            mA_fixed = np.maximum(mA_0,mB_0)
            mB_fixed = np.minimum(mA_0,mB_0)
            mA_0, mB_0 = (mA_fixed, mB_fixed)

            d0 = 10**(rand.uniform(0,np.log10(self.max_distance),size=nwalkers))
            AV0 = rand.uniform(0,self.maxAV,size=nwalkers)
            if self.fit_for_distance:
                p0 = np.array([mA_0,mB_0,age0,feh0,d0,AV0]).T
            else:
                p0 = np.array([mA_0,mB_0,age0,feh0]).T
            if initial_burn:
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike,
                                                **kwargs)
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
                p0[:,4] *= (1 + rand.normal(size=nwalkers)*0.5)
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self._sampler = sampler
        return sampler

    def triangle_plots(self, basename=None, format='png',
                       **kwargs):
        """Returns two triangle plots, one with physical params, one observational

        :param basename:
            If basename is provided, then plots will be saved as
            "[basename]_physical.[format]" and "[basename]_observed.[format]"

        :param format:
            Format in which to save figures (e.g., 'png' or 'pdf')

        :param **kwargs:
            Additional keyword arguments passed to :func:`StarModel.triangle`
            and :func:`StarModel.prop_triangle`

        :return:
             * Physical parameters triangle plot (mass_A, mass_B, radius, Teff, feh, age, distance)
             * Observed properties triangle plot.
             
        """
        fig1 = self.triangle(plot_datapoints=False,
                            params=['mass_A', 'mass_B','radius','Teff','feh','age','distance'],
                            **kwargs)
        if basename is not None:
            plt.savefig('{}_physical.{}'.format(basename,format))
            plt.close()
        fig2 = self.prop_triangle(**kwargs)
        if basename is not None:
            plt.savefig('{}_observed.{}'.format(basename,format))
            plt.close()
        return fig1, fig2

    def triangle(self, params=None, **kwargs):
        """
        Makes a nifty corner plot.

        Uses :func:`triangle.corner`.

        :param params: (optional)
            Names of columns (from :attr:`StarModel.samples`)
            to plot.  If ``None``, then it will plot samples
            of the parameters used in the MCMC fit-- that is,
            mass, age, [Fe/H], and optionally distance and A_V.

        :param query: (optional)
            Optional query on samples.

        :param extent: (optional)
            Will be appropriately passed to :func:`triangle.corner`.

        :param **kwargs:
            Additional keyword arguments passed to :func:`triangle.corner`.

        :return:
            Figure oject containing corner plot.
            
        """
        if params is None:
            params = ['mass_A', 'mass_B', 'age', 'feh', 'distance', 'AV']

        super(BinaryStarModel, self).triangle(params=params, **kwargs)


    def _make_samples(self, lnprob_thresh=0.005):
        lnprob_thresh = np.percentile(self.sampler.flatlnprobability, 0.5)
        ok = self.sampler.flatlnprobability > lnprob_thresh

        mass_A = self.sampler.flatchain[:,0][ok]
        mass_B = self.sampler.flatchain[:,1][ok]
        age = self.sampler.flatchain[:,2][ok]
        feh = self.sampler.flatchain[:,3][ok]

        if self.fit_for_distance:
            distance = self.sampler.flatchain[:,4][ok]
            AV = self.sampler.flatchain[:,5][ok]
        else:
            distance = None
            AV = 0

        df = self.ic(mass_A, age, feh, 
                       distance=distance, AV=AV)
        df_B = self.ic(mass_B, age, feh, 
                       distance=distance, AV=AV)

        for col in df_B.columns:
            if re.search('_mag', col):
                df[col] = addmags(df[col], df_B[col])

        df['mass_A'] = df['mass']
        df.drop('mass', axis=1, inplace=True)
        df['mass_B'] = df_B['mass']
        df['age'] = age
        df['feh'] = feh

        if self.fit_for_distance:
            df['distance'] = distance
            df['AV'] = AV

        self._samples = df.copy()



class TripleStarModel(StarModel):
    """Just like BinaryStarModel but for three.

    Parameters now include mass_A, mass_B, and mass_C
    """
    def loglike(self, p, use_local_fehprior=True):
        """Log-likelihood of model at given parameters

        
        :param p: 
            mass_A, mass_B, mass_C, log10(age), feh, [distance, A_V (extinction)].
            Final two should only be provided if ``self.fit_for_distance``
            is ``True``; that is, apparent magnitudes are provided.
            
        :param use_local_fehprior:
            Whether to use the Casagrande et al. (2011) prior via
            :func:`localfehdist`.  Default is ``True``.

        :return:
           log-likelihood.  Will be -np.inf if values out of range.
        
        """
        if len(p)==7:
            fit_for_distance = True
            mass_A, mass_B, mass_C, age, feh, dist, AV = p
        elif len(p)==5:
            fit_for_distance = False
            mass_A, mass_B, mass_C, age, feh = p
                        
        #keep values in range; enforce mass_A > mass_B > mass_C
        if mass_A < self.ic.minmass or mass_A > self.ic.maxmass \
           or mass_B < self.ic.minmass or mass_B > self.ic.maxmass \
           or mass_C < self.ic.minmass or mass_C > self.ic.maxmass \
           or mass_B > mass_A \
           or mass_C > mass_B or mass_C > mass_A \
           or age < self.ic.minage or age > self.ic.maxage \
           or feh < self.ic.minfeh or feh > self.ic.maxfeh:
            return -np.inf
        if fit_for_distance:
            if dist < 0 or AV < 0 or dist > self.max_distance:
                return -np.inf
            if AV > self.maxAV:
                return -np.inf

        logl = 0
        for prop in self.properties.keys():
            val,err = self.properties[prop]
            if prop in self.ic.bands:
                if not fit_for_distance:
                    raise ValueError('must fit for mass_A, mass_B, mass_C, age, feh, dist,'+ 
                                     'A_V if apparent magnitudes provided.')
                mods = self.ic.mag[prop]([mass_A, mass_B, mass_C], 
                                         age, feh) + 5*np.log10(dist) - 5
                A = AV*EXTINCTION[prop]
                mods += A
                mod = addmags(*mods)
            elif prop=='feh':
                mod = feh
            else:
                mod = getattr(self.ic,prop)(mass_A,age,feh)
            logl += -(val-mod)**2/err**2

        if np.isnan(logl):
            logl = -np.inf


        #IMF prior
        logl += np.log(salpeter_prior(mass_A))
        
        #distance prior ~d^2 out to d_max
        if fit_for_distance:
            logl += np.log(3/self.max_distance**3 * dist**2)

        if use_local_fehprior:
            #From Jo Bovy:
            #https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
            #2D gaussian fit based on Casagrande (2011)

            fehdist= 0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
                +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.)
            logl += np.log(fehdist)

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

        :return:
            list of best-fit parameters: ``[mA,mB,age,feh,[distance,A_V]]``.
            Note that distance and A_V values will be meaningless unless
            magnitudes are present in ``self.properties``.
        
        """
        mA_0,age0,feh0 = self.ic.random_points(nseeds)
        mB_0,foo1,foo2 = self.ic.random_points(nseeds)
        mC_0,foo3,foo4 = self.ic.random_points(nseeds)
        m_all = np.sort(np.array([mA_0, mB_0, mC_0]), axis=0)
        mA_0, mB_0, mC_0 = (m_all[0,:], m_all[1,:], m_all[2,:])

        d0 = 10**(rand.uniform(0,np.log10(self.max_distance),size=nseeds))
        AV0 = rand.uniform(0,self.maxAV,size=nseeds)

        

        costs = np.zeros(nseeds)

        if self.fit_for_distance:
            pfits = np.zeros((nseeds,7))
        else:
            pfits = np.zeros((nseeds,5))
            
        def fn(p): #fmin is a function *minimizer*
            return -1*self.loglike(p)
        
        for i,mA,mB,mC,age,feh,d,AV in zip(range(nseeds),
                                    mA_0,mB_0,mC_0,age0,feh0,d0,AV0):
                if self.fit_for_distance:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,mC,age,feh,d,AV],disp=False)
                else:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,mC,age,feh],disp=False)
                pfits[i,:] = pfit
                costs[i] = self.loglike(pfit)

        return pfits[np.argmax(costs),:]

            
    def fit_mcmc(self,nwalkers=200,nburn=100,niter=200,
                 p0=None,initial_burn=None,
                 ninitial=100, loglike_kwargs=None,
                 **kwargs):
        """Fits stellar model using MCMC.

        See :func:`StarModel.fit_mcmc`.
            
        """

        #clear any saved _samples
        if self._samples is not None:
            self._samples = None
            

        if self.fit_for_distance:
            npars = 7
            if initial_burn is None:
                initial_burn = True
        else:
            if initial_burn is None:
                initial_burn = False
            npars = 5

        if p0 is None:
            mA_0,age0,feh0 = self.ic.random_points(nwalkers)
            mB_0,foo1,foo2 = self.ic.random_points(nwalkers)
            mC_0,foo3,foo4 = self.ic.random_points(nwalkers)
            m_all = np.sort(np.array([mA_0, mB_0, mC_0]), axis=0)
            mA_0, mB_0, mC_0 = (m_all[0,:], m_all[1,:], m_all[2,:])

            d0 = 10**(rand.uniform(0,np.log10(self.max_distance),size=nwalkers))
            AV0 = rand.uniform(0,self.maxAV,size=nwalkers)
            if self.fit_for_distance:
                p0 = np.array([mA_0,mB_0,mC_0,age0,feh0,d0,AV0]).T
            else:
                p0 = np.array([mA_0,mB_0,mC_0,age0,feh0]).T
            if initial_burn:
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike,
                                                **kwargs)
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
                p0[:,5] *= (1 + rand.normal(size=nwalkers)*0.5) #distance
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.loglike)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self._sampler = sampler
        return sampler

    def triangle_plots(self, basename=None, format='png',
                       **kwargs):
        """Returns two triangle plots, one with physical params, one observational

        :return:
             * Physical parameters triangle plot (mass_A, mass_B, mass_C, radius, 
                Teff, feh, age, distance)
             * Observed properties triangle plot.
             
        """
        fig1 = self.triangle(plot_datapoints=False,
                            params=['mass_A', 'mass_B', 'mass_C', 'radius',
                                    'Teff','feh','age','distance'],
                            **kwargs)
        if basename is not None:
            plt.savefig('{}_physical.{}'.format(basename,format))
            plt.close()
        fig2 = self.prop_triangle(**kwargs)
        if basename is not None:
            plt.savefig('{}_observed.{}'.format(basename,format))
            plt.close()
        return fig1, fig2

    def triangle(self, params=None, **kwargs):
        """
        Makes a nifty corner plot.

        """
        if params is None:
            params = ['mass_A', 'mass_B', 'mass_C', 
                      'age', 'feh', 'distance', 'AV']

        super(TripleStarModel, self).triangle(params=params, **kwargs)


    def _make_samples(self, lnprob_thresh=0.005):
        
        lnprob_thresh = np.percentile(self.sampler.flatlnprobability, 
                                      lnprob_thresh*100)
        ok = self.sampler.flatlnprobability > lnprob_thresh

        mass_A = self.sampler.flatchain[:,0][ok]
        mass_B = self.sampler.flatchain[:,1][ok]
        mass_C = self.sampler.flatchain[:,2][ok]
        age = self.sampler.flatchain[:,3][ok]
        feh = self.sampler.flatchain[:,4][ok]

        if self.fit_for_distance:
            distance = self.sampler.flatchain[:,5][ok]
            AV = self.sampler.flatchain[:,6][ok]
        else:
            distance = None
            AV = 0

        df = self.ic(mass_A, age, feh, 
                       distance=distance, AV=AV)
        df_B = self.ic(mass_B, age, feh, 
                       distance=distance, AV=AV)
        df_C = self.ic(mass_C, age, feh, 
                       distance=distance, AV=AV)

        for col in df_B.columns:
            if re.search('_mag', col):
                df[col] = addmags(df[col], df_B[col], df_C[col])

        df['mass_A'] = df['mass']
        df.drop('mass', axis=1, inplace=True)
        df['mass_B'] = df_B['mass']
        df['mass_C'] = df_C['mass']
        df['age'] = age
        df['feh'] = feh

        if self.fit_for_distance:
            df['distance'] = distance
            df['AV'] = AV

        self._samples = df.copy()

class MultipleStarModel(StarModel, BinaryStarModel, TripleStarModel):
    """
    StarModel where N_stars (1,2, or 3) is a parameter to estimate.
    """

    def binary_loglike(self, *args, **kwargs):
        return BinaryStarModel.loglike(self, *args, **kwargs)

    def triple_loglike(self, *args, **kwargs):
        return TripleStarModel.loglike(self, *args, **kwargs)
        

#### Utility functions #####


def addmags(*mags):
    tot=0
    for mag in mags:
        tot += 10**(-0.4*mag)
    return -2.5*np.log10(tot)
    

def salpeter_prior(m,alpha=-2.35,minmass=0.1,maxmass=10):
    C = (1+alpha)/(maxmass**(1+alpha)-minmass**(1+alpha))
    if m < minmass or m > maxmass:
        return 0
    else:
        return C*m**(alpha)

