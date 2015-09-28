from __future__ import print_function, division
import os,os.path, glob
import logging
import re
import json

from configobj import ConfigObj

try:
    import numpy as np
except ImportError:
    np = None

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
    
try:
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
except ImportError:
    plt = None
    gaussian_kde = None

try:
    import triangle
except ImportError:
    triangle = None

mnest_available = True
try:
    import pymultinest
except ImportError:
    logging.warning('PyMultiNest not available; only emcee fits will be possible.')
    pymultinest = None
    mnest_available = False


from .extinction import EXTINCTION
from .passbands import WEFF

class StarModel(object):
    """An object to represent a star, with observed properties, modeled by an Isochrone

    This is used to fit a physical stellar model to observed
    quantities, e.g. spectroscopic or photometric, based on
    an :class:`Isochrone`.  Parallax (in miliarcseconds) is
    also accepted as an observed quantity.

    Note that by default a local metallicity prior, based on SDSS data,
    will be used when :func:`StarModel.fit` is called.

    :param ic: 
        :class:`Isochrone` object used to model star.

    :param maxAV: (optional)
        Maximum allowed extinction (i.e. the extinction @ infinity in direction of star).  Default is 1.

    :param max_distance: (optional)
        Maximum allowed distance (pc).  Default is 3000.
    
    :param use_emcee: (optional)
        If set to true, then sampling done with emcee rather than MultiNest.
        (not recommended unless you have very precise spectroscopic properties).

    :param **kwargs:
        Keyword arguments must be properties of given isochrone, e.g., logg,
        feh, Teff, and/or magnitudes.  The values represent measurements of
        the star, and must be in (value,error) format. All such keyword
        arguments will be held in ``self.properties``.  ``parallax`` is
        also a valid property, and should be provided in miliarcseconds.
        
    """
    def __init__(self,ic,maxAV=1,max_distance=3000,
                 use_emcee=False, 
                 min_logg=None, name='',
                 **kwargs):
        self._ic = ic
        self.properties = kwargs
        self.max_distance = max_distance
        self.maxAV = maxAV

        self.name = name

        self._samples = None
        self._mnest_samples = None
        self.use_emcee = use_emcee
        if not mnest_available:
            logging.warning('MultiNest not available; use_emcee being set to True')
            self.use_emcee = True

        self.min_logg = min_logg
            
        self.n_params = 5 #mass, feh, age, distance, AV
        
        self._props_cleaned = False
        self._mnest_basename = None


    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    @classmethod
    def from_ini(cls, ic, folder='.', ini_file='star.ini'):
        """
        Initialize a StarModel from a .ini file

        File should contain all arguments with which to initialize
        StarModel.  
        """
        if not os.path.isabs(ini_file):
            ini_file = os.path.join(folder,ini_file)

        config = ConfigObj(ini_file)
        kwargs = {}
        for kw in config.keys():
            try:
                kwargs[kw] = float(config[kw])
            except:
                kwargs[kw] = (float(config[kw][0]), float(config[kw][1]))

        return cls(ic, **kwargs)
    
    @property
    def mags(self):
        d = {}
        for prop,vals in self.properties.items():
            if prop in self.ic.bands:
                try:
                    val,err = vals
                except TypeError:
                    val = vals
                d[prop] = val
        return d

    @property
    def mag_errs(self):
        d = {}
        for prop,vals in self.properties.items():
            if prop in self.ic.bands:
                try:
                    val,err = vals
                except TypeError:
                    continue
                d[prop] = err
        return d
    
    @property
    def Teff(self):
        if 'Teff' in self.properties:
            return self.properties['Teff']

    @property
    def feh(self):
        if 'feh' in self.properties:
            return self.properties['feh']

    @property
    def logg(self):
        if 'logg' in self.properties:
            return self.properties['logg']


    def _clean_props(self):
        """
        Makes sure all properties are legit for isochrone.

        Not done in __init__ in order to save speed on loading.
        """
        remove = []
        for p in self.properties.keys():
            if not hasattr(self.ic, p) and \
              p not in self.ic.bands and p not in ['parallax','feh','age','mass_B','mass_C'] and \
              not re.search('delta_',p):
                remove.append(p)

        for p in remove:
            del self.properties[p]

        if len(remove) > 0:
            logging.warning('Properties removed from Model because ' +
                            'not present in {}: {}'.format(type(self.ic),remove))

        remove = []
        for p in self.properties.keys():
            try:
                val = self.properties[p][0]
                if not np.isfinite(val):
                    remove.append(p)
            except:
                pass

        for p in remove:
            del self.properties[p]

        if len(remove) > 0:
            logging.warning('Properties removed from Model because ' +
                            'value is nan or inf: {}'.format(remove))
        


        self._props_cleaned = True
    
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
            
    def loglike(self, *args, **kwargs):
        """For backwards compatibility
        """
        return lnpost(*args, **kwargs)

    def lnlike(self, p):
        """Log-likelihood of model at given parameters

        
        :param p: 
            mass, log10(age), feh, [distance, A_V (extinction)].
            Final two should only be provided if ``self.fit_for_distance``
            is ``True``; that is, apparent magnitudes are provided.
            
        :return:
           log-likelihood.  Will be -np.inf if values out of range.
        
        """

        if not self._props_cleaned:
            self._clean_props()
            
        if not self.use_emcee:
            fit_for_distance = True
            mass, age, feh, dist, AV = (p[0], p[1], p[2], p[3], p[4])
        else:
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

        if self.min_logg is not None:
            logg = self.ic.logg(mass,age,feh)
            if logg < self.min_logg:
                return -np.inf

        logl = 0
        for prop in self.properties.keys():
            try:
                val,err = self.properties[prop]
            except TypeError:
                #property not appropriate for fitting (e.g. no error provided)
                continue

            if prop in self.ic.bands:
                if not fit_for_distance:
                    raise ValueError('must fit for mass, age, feh, dist, A_V if apparent magnitudes provided.')
                mod = self.ic.mag[prop](mass,age,feh) + 5*np.log10(dist) - 5
                A = AV*EXTINCTION[prop]
                mod += A
            elif re.search('delta_',prop):
                continue
            elif prop=='feh':
                mod = feh
            elif prop=='parallax':
                mod = 1./dist * 1000
            else:
                mod = getattr(self.ic,prop)(mass,age,feh)

            logl += -(val-mod)**2/(2*err**2) + np.log(1/(err*np.sqrt(2*np.pi)))

        if np.isnan(logl):
            logl = -np.inf

        return logl

    def lnprior(self, mass, age, feh,
                distance=None, AV=None, 
                use_local_fehprior=True):
        """
        log-prior for model parameters
        
        """
        mass_prior = salpeter_prior(mass)
        if mass_prior==0:
            mass_lnprior = -np.inf
        else:
            mass_lnprior = np.log(mass_prior)

        if np.isnan(mass_lnprior):
            logging.warning('mass prior is nan at {}'.format(mass))

        age_lnprior = np.log(age * (2/(self.ic.maxage**2-self.ic.minage**2)))
        if np.isnan(age_lnprior):
            logging.warning('age prior is nan at {}'.format(age))


        if use_local_fehprior:
            fehdist = local_fehdist(feh)
        else:
            fehdist = 1/(self.ic.maxfeh - self.ic.minfeh)
        feh_lnprior = np.log(fehdist)
        if np.isnan(feh_lnprior):
            logging.warning('feh prior is nan at {}'.format(feh))


        if distance is not None:
            if distance <= 0:
                distance_lnprior = -np.inf
            else:
                distance_lnprior = np.log(3/self.max_distance**3 * distance**2)
        else:
            distance_lnprior = 0
        if np.isnan(distance_lnprior):
            logging.warning('distance prior is nan at {}'.format(distance))


        if AV is not None:
            AV_lnprior = np.log(1/self.maxAV)
        else:
            AV_lnprior = 0
        if np.isnan(AV_lnprior):
            logging.warning('AV prior is nan at {}'.format(AV))
            

        lnprior = (mass_lnprior + age_lnprior + feh_lnprior + 
                distance_lnprior + AV_lnprior)

        return lnprior

    def lnpost(self, p, use_local_fehprior=True):
        """
        log-posterior of model at given parameters
        """
        if not self.use_emcee:
            mass, age, feh, dist, AV = (p[0], p[1], p[2], p[3], p[4])
        else:
            if len(p)==5:
                fit_for_distance = True
                mass,age,feh,dist,AV = p
            elif len(p)==3:
                fit_for_distance = False
                mass,age,feh = p
                dist = None
                AV = None
            
        return (self.lnlike(p) + 
                self.lnprior(mass, age, feh, dist, AV,
                             use_local_fehprior=use_local_fehprior))



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
            return -1*self.lnpost(p)
        
        for i,m,age,feh,d,AV in zip(range(nseeds),
                                    m0,age0,feh0,d0,AV0):
                if self.fit_for_distance:
                    pfit = scipy.optimize.fmin(fn,[m,age,feh,d,AV],disp=False)
                else:
                    pfit = scipy.optimize.fmin(fn,[m,age,feh],disp=False)
                pfits[i,:] = pfit
                costs[i] = self.lnpost(pfit)

        return pfits[np.argmax(costs),:]

    def mnest_prior(self, cube, ndim, nparams):
        """
        Transforms unit cube into parameter cube.

        Parameters if running multinest must be mass, age, feh, distance, AV.
        """
        cube[0] = (self.ic.maxmass - self.ic.minmass)*cube[0] + self.ic.minmass
        cube[1] = (self.ic.maxage - self.ic.minage)*cube[1] + self.ic.minage
        cube[2] = (self.ic.maxfeh - self.ic.minfeh)*cube[2] + self.ic.minfeh
        cube[3] = cube[3]*self.max_distance
        cube[4] = cube[4]*self.maxAV
    
    def mnest_loglike(self, cube, ndim, nparams):
        """loglikelihood function for multinest
        """
        return self.lnpost(cube)

    def fit(self, **kwargs):
        """
        Wrapper for either :func:`fit_multinest` or :func:`fit_mcmc`.

        Default will be to use MultiNest; set `use_emcee` keyword to `True` 
        if you want to use MCMC, or just call :func:`fit_mcmc` directly.
        """
        if self.use_emcee:
            if 'basename' in kwargs:
                del kwargs['basename']
            if 'verbose' in kwargs:
                del kwargs['verbose']
            if 'overwrite' in kwargs:
                del kwargs['overwrite']
            self.fit_mcmc(**kwargs)
        else:
            self.fit_multinest(**kwargs)

    def fit_multinest(self, n_live_points=1000, basename='chains/single-',
                      verbose=True, refit=False, overwrite=False,
                      **kwargs):
        """
        Fits model using MultiNest, via pymultinest.  

        :param n_live_points:
            Number of live points to use for MultiNest fit.

        :param basename:
            Where the MulitNest-generated files will live.  
            By default this will be in a folder named `chains`
            in the current working directory.  Calling this 
            will define a `_mnest_basename` attribute for 
            this object.

        :param verbose:
            Whether you want MultiNest to talk to you.

        :param refit, overwrite:
            Set either of these to true if you want to 
            delete the MultiNest files associated with the
            given basename and start over.

        :param **kwargs:
            Additional keyword arguments will be passed to 
            :func:`pymultinest.run`.

        """
        folder = os.path.abspath(os.path.dirname(basename))
        if not os.path.exists(folder):
            os.makedirs(folder)

        #If previous fit exists, see if it's using the same
        # observed properties
        prop_nomatch = False
        propfile = '{}properties.json'.format(basename)
        if os.path.exists(propfile):
            with open(propfile) as f:
                props = json.load(f)
            if set(props.keys()) != set(self.properties.keys()):
                prop_nomatch = True
            else:
                for k,v in props.items():
                    if np.size(v)==2:
                        if not self.properties[k][0] == v[0] and \
                                self.properties[k][1] == v[1]:
                            props_nomatch = True
                    else:
                        if not self.properties[k] == v:
                            props_nomatch = True

        if prop_nomatch and not overwrite:
            raise ValueError('Properties not same as saved chains ' +
                            '(basename {}*). '.format(basename) +
                            'Use overwrite=True to fit.')


        if refit or overwrite:
            files = glob.glob('{}*'.format(basename))
            [os.remove(f) for f in files]

        self._mnest_basename = basename

        pymultinest.run(self.mnest_loglike, self.mnest_prior, self.n_params,
                        n_live_points=n_live_points, outputfiles_basename=basename,
                        verbose=verbose,
                        **kwargs)

        with open(propfile, 'w') as f:
            json.dump(self.properties, f, indent=2)

        self._make_samples()

    @property
    def mnest_analyzer(self):
        """
        PyMultiNest Analyzer object associated with fit.  

        See PyMultiNest documentation for more.
        """
        return pymultinest.Analyzer(self.n_params, self._mnest_basename)

    @property
    def evidence(self):
        """
        Log(evidence) from multinest fit
        """
        s = self.mnest_analyzer.get_stats()
        return (s['global evidence'],s['global evidence error'])

    def fit_mcmc(self,nwalkers=300,nburn=200,niter=100,
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

        :param **kwargs:
            Additional keyword arguments passed to :class:`emcee.EnsembleSampler`
            constructor.
            
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
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost,
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
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self._sampler = sampler
        return sampler

    def mag_plot(self, height=500, pix_width=20, spacing=20,
                 edge=0.1, figsize=(8,6)):
        
        bands = np.array(self.mag_errs.keys())
        weffs = np.array([WEFF[b] for b in bands])
        inds = np.argsort(weffs)
        bands = bands[inds]
        weffs = weffs[inds]

        q = 0.01
        minmag = min(np.min([self.samples['{}_mag'.format(b)].quantile(q) 
                             for b in bands]) - edge,
                     np.min([self.properties[b][0] - edge for b in bands]))
        maxmag = max(np.max([self.samples['{}_mag'.format(b)].quantile(1-q) 
                             for b in bands]) + edge,
                     np.max([self.properties[b][0] + edge for b in bands]))

        n_bands = len(bands)
        width = n_bands * (pix_width + spacing) + spacing
        mag_grid = np.linspace(minmag, maxmag, height)[::-1]

        image = np.zeros((height, width))

        plt.figure(figsize=figsize)

        mids = []
        for i,b in enumerate(bands):
            col1 = spacing*(i+1) + i*(pix_width)
            col2 = spacing*(i+1) + (i+1)*(pix_width)
            mids.append((col1 + col2)//2)
            vslice = image[:, col1:col2]

            kde = gaussian_kde(self.samples['{}_mag'.format(b)])
            pdf = kde(mag_grid)
            vslice += pdf[:, np.newaxis]

        extent = [0, image.shape[1], maxmag, minmag]
        plt.imshow(image, aspect='auto', cmap='binary', 
                   extent=extent, origin='lower')
        ax = plt.gca()
        ax.set_xticks(mids)
        ax.set_xticklabels(bands, fontsize=18);
        ax.set_ylabel('mag', fontsize=18)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[::-1])
        plt.tick_params(axis='y', labelsize=16)

        for i,(b,m) in enumerate(zip(bands,mids)):
            val, err = self.properties[b]
            plt.errorbar(m, val, err, marker='o', color='w', 
                         ms=4, lw=5, mec='w', mew=5)    
            plt.errorbar(m, val, err, marker='o', color='r', 
                         ms=4, lw=3, mec='r', mew=3)

        plt.title(self.name, fontsize=20)

        return plt.gcf()

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
                                 params=['mass','radius','Teff','logg','feh','age',
                                         'distance','AV'],
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
        remove = []
        for i,par in enumerate(params):
            m = re.search('delta_(\w+)$',par)
            if m:
                if type(self) == BinaryStarModel:
                    b = m.group(1)
                    values = (df['{}_mag_B'.format(b)] - 
                              df['{}_mag_A'.format(b)])
                    df[par] = values
                else:
                    remove.append(i)
                    continue
                    
            else:
                values = df[par]
            qs = np.array([0.5 - 0.5*extent, 0.5 + 0.5*extent])
            minval, maxval = values.quantile(qs)
            if 'truths' in kwargs:
                datarange = maxval - minval
                if kwargs['truths'][i] < minval:
                    minval = kwargs['truths'][i] - 0.05*datarange
                if kwargs['truths'][i] > maxval:
                    maxval = kwargs['truths'][i] + 0.05*datarange
            extents.append((minval,maxval))
            
        [params.pop(i) for i in remove]

        fig = triangle.corner(df[params], labels=params, 
                               extents=extents, **kwargs)

        fig.suptitle(self.name, fontsize=22)
        return fig

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
            try:
                val, err = self.properties[p]
            except:
                continue

            if p in self.ic.bands:
                params.append('{}_mag'.format(p))
                truths.append(val)
            elif p=='parallax':
                params.append('distance')
                truths.append(1/(val/1000.))
            else:
                params.append(p)
                truths.append(val)
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

    def _make_samples(self):

        if not self.use_emcee:
            chain = np.loadtxt('{}post_equal_weights.dat'.format(self._mnest_basename))

            #for purposes of unit test, sometimes there will be 1-length chain...
            if chain.ndim==1:
                chain = np.array([chain])

            mass = chain[:,0]
            age = chain[:,1]
            feh = chain[:,2]
            distance = chain[:,3]
            AV = chain[:,4]
            lnprob = chain[:,-1]

        else:
            #select out only walkers with > 0.15 acceptance fraction
            ok_walkers = self.sampler.acceptance_fraction > 0.15

            mass = self.sampler.chain[ok_walkers, :, 0].ravel()
            age = self.sampler.chain[ok_walkers, :, 1].ravel()
            feh = self.sampler.chain[ok_walkers, :, 2].ravel()

            if self.fit_for_distance:
                distance = self.sampler.chain[ok_walkers, :, 3].ravel()
                AV = self.sampler.chain[ok_walkers, :, 4].ravel()
            else:
                distance = None
                AV = 0

            lnprob = self.sampler.lnprobability[ok_walkers, :].ravel()

        df = self.ic(mass, age, feh, 
                     distance=distance, AV=AV)
        df['age'] = age
        df['feh'] = feh

        if self.fit_for_distance:
            df['distance'] = distance
            df['AV'] = AV

        df['lnprob'] = lnprob

        self._samples = df.copy()
        
    @property
    def samples(self):
        """Dataframe with samples drawn from isochrone according to posterior

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
            return samples, (med,lo,hi)
        else:
            return samples

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
        attrs.min_logg = self.min_logg

        attrs.use_emcee = self.use_emcee
        attrs._mnest_basename = self._mnest_basename

        attrs.name = self.name

        store.close()

    @classmethod
    def load_hdf(cls, filename, path='', name=None):
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
        min_logg = attrs.min_logg
        ic_type = attrs.ic_type
        use_emcee = attrs.use_emcee
        basename = attrs._mnest_basename
        
        if name is None:
            try:
                name = attrs.name
            except:
                name = ''

        store.close()

        #ic = ic_type() don't need to initialize anymore

        mod = cls(ic_type, maxAV=maxAV, max_distance=max_distance,
                  use_emcee=use_emcee, name=name,
                  **properties)
        mod._samples = samples
        mod._mnest_basename = basename
        return mod


class BinaryStarModel(StarModel):
    """
    Object used to fit two stars at the same distance to given observed properties

    Initialize the same way as :class:`StarModel`. 

    Difference between this object and a regular :class:`StarModel` is that
    the fit parameters include two masses: ``mass_A`` and ``mass_B`` instead
    of just one.

    Notably, this object can also take additional 
    ``delta_mag`` properties, representing contrast measurements of a companion
    star.  These should be called, e.g., ``delta_r``.  This will be an additional
    constraint in the model fitting.  All the other provided apparent magnitudes
    should be the total combined light of the two stars.




    """
    def __init__(self, *args, **kwargs):
        super(BinaryStarModel, self).__init__(*args, **kwargs)

        self.n_params = 6

    def lnlike(self, p):
        """Log-likelihood of model at given parameters

        
        :param p: 
            mass_A, mass_B, log10(age), feh, [distance, A_V (extinction)].
            Final two should only be provided if ``self.fit_for_distance``
            is ``True``; that is, apparent magnitudes are provided.
            
        :return:
           log-likelihood.  Will be -np.inf if values out of range.
        
        """
        if not self._props_cleaned:
            self._clean_props()

        if not self.use_emcee:
            fit_for_distance = True
            mass_A, mass_B, age, feh, dist, AV = (p[0], p[1], p[2],
                                                  p[3], p[4], p[5])
        else:
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

        if self.min_logg is not None:
            logg = self.ic.logg(mass_A,age,feh)
            if logg < self.min_logg:
                return -np.inf        

        logl = 0
        for prop in self.properties.keys():
            try:
                val,err = self.properties[prop]
            except TypeError:
                #property not appropriate for fitting (e.g. no error provided)
                continue
            m = re.search('delta_(\w+)',prop)
            if prop in self.ic.bands:
                if not fit_for_distance:
                    raise ValueError('must fit for mass, age, feh, dist, A_V '+
                                     'if apparent magnitudes provided.')
                mod_A = self.ic.mag[prop](mass_A,age,feh) + 5*np.log10(dist) - 5
                mod_B = self.ic.mag[prop](mass_B,age,feh) + 5*np.log10(dist) - 5
                A = AV*EXTINCTION[prop]
                mod_A += A
                mod_B += A
                mod = addmags(mod_A, mod_B)
            elif m:
                band = m.group(1)
                mod_A = self.ic.mag[band](mass_A,age,feh) + 5*np.log10(dist) - 5
                mod_B = self.ic.mag[band](mass_B,age,feh) + 5*np.log10(dist) - 5
                A = AV*EXTINCTION[band]
                mod_A += A
                mod_B += A
                mod = mod_B - mod_A
            elif prop=='feh':
                mod = feh
            elif prop=='mass_B':
                mod = mass_B
            elif prop=='parallax':
                mod = 1./dist * 1000
            else:
                mod = getattr(self.ic,prop)(mass_A,age,feh)

            logl += -(val-mod)**2/(2*err**2) + np.log(1/(err*np.sqrt(2*np.pi)))


        if np.isnan(logl):
            logl = -np.inf


        return logl
        
    def lnprior(self, mass_A, mass_B, age, feh, 
                distance=None, AV=None, use_local_fehprior=True):
        lnpr = super(BinaryStarModel,self).lnprior(mass_A, age, feh, distance, AV,
                                                  use_local_fehprior=use_local_fehprior)
        q = mass_B / mass_A
        lnpr += np.log(q_prior(q, mass_A))
        return lnpr

    def lnpost(self, p, use_local_fehprior=True):
        if not self.use_emcee:
            mass_A,mass_B,age,feh,dist,AV = (p[0], p[1], p[2],
                                             p[3], p[4], p[5])
        else:
            if len(p)==6:
                fit_for_distance = True
                mass_A,mass_B,age,feh,dist,AV = p
            elif len(p)==4:
                fit_for_distance = False
                mass_A,mass_B,age,feh = p
                dist = None
                AV = None

        return (self.lnlike(p) + 
                self.lnprior(mass_A, mass_B, age, feh, dist, AV,
                             use_local_fehprior=use_local_fehprior))
        

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
            return -1*self.lnpost(p)
        
        for i,mA,mB,age,feh,d,AV in zip(range(nseeds),
                                    mA_0,mB_0,age0,feh0,d0,AV0):
                if self.fit_for_distance:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,age,feh,d,AV],disp=False)
                else:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,age,feh],disp=False)
                pfits[i,:] = pfit
                costs[i] = self.lnpost(pfit)

        return pfits[np.argmax(costs),:]

    def mnest_prior(self, cube, ndim, nparams):
        cube[0] = (self.ic.maxmass - self.ic.minmass)*cube[0] + self.ic.minmass
        cube[1] = (cube[0] - self.ic.minmass)*cube[1] + self.ic.minmass
        cube[2] = (self.ic.maxage - self.ic.minage)*cube[2] + self.ic.minage
        cube[3] = (self.ic.maxfeh - self.ic.minfeh)*cube[3] + self.ic.minfeh
        cube[4] = cube[4]*self.max_distance
        cube[5] = cube[5]*self.maxAV
        
    def fit_multinest(self, basename='chains/binary-', **kwargs):
        super(BinaryStarModel, self).fit_multinest(basename=basename, **kwargs)

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
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost,
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
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost)
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
                            params=['mass_A', 'mass_B','radius','Teff','logg','feh','age',
                                    'distance', 'AV'],
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


    def _make_samples(self):

        if not self.use_emcee:
            chain = np.loadtxt('{}post_equal_weights.dat'.format(self._mnest_basename))

            #for purposes of unit test, sometimes there will be 1-length chain...
            if chain.ndim==1:
                chain = np.array([chain])

            mass_A = chain[:,0]
            mass_B = chain[:,1]
            age = chain[:,2]
            feh = chain[:,3]
            distance = chain[:,4]
            AV = chain[:,5]
            lnprob = chain[:,-1]

        else:
            #select out legit walkers
            ok_walkers = self.sampler.acceptance_fraction > 0.15

            mass_A = self.sampler.chain[ok_walkers,:,0].ravel()
            mass_B = self.sampler.chain[ok_walkers,:,1].ravel()
            age = self.sampler.chain[ok_walkers,:,2].ravel()
            feh = self.sampler.chain[ok_walkers,:,3].ravel()

            if self.fit_for_distance:
                distance = self.sampler.chain[ok_walkers,:,4].ravel()
                AV = self.sampler.chain[ok_walkers,:,5].ravel()
            else:
                distance = None
                AV = 0
            lnprob = self.sampler.lnprobability[ok_walkers,:].ravel()


        df = self.ic(mass_A, age, feh, 
                       distance=distance, AV=AV)
        df_B = self.ic(mass_B, age, feh, 
                       distance=distance, AV=AV)

        for col in df_B.columns:
            m = re.search('_mag$', col)
            if m:
                df['{}_A'.format(col)] = df[col]
                df['{}_B'.format(col)] = df_B[col]
                df[col] = addmags(df[col], df_B[col])


        df['mass_A'] = df['mass']
        df.drop('mass', axis=1, inplace=True)
        df['mass_B'] = df_B['mass']
        df['radius_B'] = df_B['radius']
        df['Teff_B'] = df_B['Teff']
        df['logg_B'] = df_B['logg']
        df['logL_B'] = df_B['logL']
        df['age'] = age
        df['feh'] = feh

        if self.fit_for_distance:
            df['distance'] = distance
            df['AV'] = AV

        df['lnprob'] = lnprob

        self._samples = df.copy()



class TripleStarModel(StarModel):
    """Just like BinaryStarModel but for three.

    Parameters now include mass_A, mass_B, and mass_C
    """
    def __init__(self, *args, **kwargs):
        super(TripleStarModel, self).__init__(*args, **kwargs)

        self.n_params = 7

    def lnlike(self, p):
        """Log-likelihood of model at given parameters

        
        :param p: 
            mass_A, mass_B, mass_C, log10(age), feh, [distance, A_V (extinction)].
            Final two should only be provided if ``self.fit_for_distance``
            is ``True``; that is, apparent magnitudes are provided.
            
        :return:
           log-likelihood.  Will be -np.inf if values out of range.
        
        """
        if not self._props_cleaned:
            self._clean_props()

        if not self.use_emcee:
            fit_for_distance = True
            mass_A, mass_B, mass_C, age, feh, dist, AV = (p[0], p[1], p[2],
                                                          p[3], p[4], p[5],
                                                          p[6])
        else:
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

        if self.min_logg is not None:
            logg = self.ic.logg(mass_A,age,feh)
            if logg < self.min_logg:
                return -np.inf

        logl = 0
        for prop in self.properties.keys():
            try:
                val,err = self.properties[prop]
            except TypeError:
                #property not appropriate for fitting (e.g. no error provided)
                continue
            if prop in self.ic.bands:
                if not fit_for_distance:
                    raise ValueError('must fit for mass_A, mass_B, mass_C, age, feh, dist,'+ 
                                     'A_V if apparent magnitudes provided.')
                mods = self.ic.mag[prop]([mass_A, mass_B, mass_C], 
                                         age, feh) + 5*np.log10(dist) - 5
                A = AV*EXTINCTION[prop]
                mods += A
                mod = addmags(*mods)
            elif re.search('delta_',prop):
                continue
            elif prop=='feh':
                mod = feh
            elif prop=='mass_B':
                mod = mass_B
            elif prop=='mass_C':
                mod = mass_C
            elif prop=='parallax':
                mod = 1./dist * 1000
            else:
                mod = getattr(self.ic,prop)(mass_A,age,feh)

            logl += -(val-mod)**2/(2*err**2) + np.log(1/(err*np.sqrt(2*np.pi)))



        if np.isnan(logl):
            logl = -np.inf

        return logl

    def lnprior(self, mass_A, mass_B, mass_C, age, feh, 
                distance=None, AV=None, use_local_fehprior=True):
        lnpr = super(TripleStarModel,self).lnprior(mass_A, age, feh, distance, AV,
                                                  use_local_fehprior=use_local_fehprior)
        q1 = mass_B / mass_A
        q2 = mass_C / mass_B
        lnpr += np.log(q_prior(q1, mass_A))
        lnpr += np.log(q_prior(q2, mass_B))
        return lnpr


    def lnpost(self, p, use_local_fehprior=True):
        if not self.use_emcee:
            mass_A,mass_B,mass_C,age,feh,dist,AV = (p[0], p[1], p[2],
                                                    p[3], p[4], p[5], 
                                                    p[6])
        else:
            if len(p)==7:
                fit_for_distance = True
                mass_A,mass_B,mass_C,age,feh,dist,AV = p
            elif len(p)==5:
                fit_for_distance = False
                mass_A,mass_B,mass_C,age,feh = p
                dist = None
                AV = None

        return (self.lnlike(p) + 
                self.lnprior(mass_A, mass_B, mass_C, age, feh, dist, AV,
                             use_local_fehprior=use_local_fehprior))
        
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
            return -1*self.lnpost(p)
        
        for i,mA,mB,mC,age,feh,d,AV in zip(range(nseeds),
                                    mA_0,mB_0,mC_0,age0,feh0,d0,AV0):
                if self.fit_for_distance:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,mC,age,feh,d,AV],disp=False)
                else:
                    pfit = scipy.optimize.fmin(fn,[mA,mB,mC,age,feh],disp=False)
                pfits[i,:] = pfit
                costs[i] = self.lnpost(pfit)

        return pfits[np.argmax(costs),:]

            
    def mnest_prior(self, cube, ndim, nparams):
        cube[0] = (self.ic.maxmass - self.ic.minmass)*cube[0] + self.ic.minmass
        cube[1] = (cube[0] - self.ic.minmass)*cube[1] + self.ic.minmass
        cube[2] = (cube[1] - self.ic.minmass)*cube[2] + self.ic.minmass
        cube[3] = (self.ic.maxage - self.ic.minage)*cube[3] + self.ic.minage
        cube[4] = (self.ic.maxfeh - self.ic.minfeh)*cube[4] + self.ic.minfeh
        cube[5] = cube[5]*self.max_distance
        cube[6] = cube[6]*self.maxAV
        
    def fit_multinest(self, basename='chains/triple-', **kwargs):
        super(TripleStarModel, self).fit_multinest(basename=basename, **kwargs)    

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
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost,
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
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost)
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
                                    'Teff','logg','feh','age','distance','AV'],
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


    def _make_samples(self):
        
        if not self.use_emcee:
            chain = np.loadtxt('{}post_equal_weights.dat'.format(self._mnest_basename))

            #for purposes of unit test, sometimes there will be 1-length chain...
            if chain.ndim==1:
                chain = np.array([chain])

            mass_A = chain[:,0]
            mass_B = chain[:,1]
            mass_C = chain[:,2]
            age = chain[:,3]
            feh = chain[:,4]
            distance = chain[:,5]
            AV = chain[:,6]
            lnprob = chain[:,-1]

        else:
            ok_walkers = self.sampler.acceptance_fraction > 0.15

            mass_A = self.sampler.chain[ok_walkers,:,0].ravel()
            mass_B = self.sampler.chain[ok_walkers,:,1].ravel()
            mass_C = self.sampler.chain[ok_walkers,:,2].ravel()
            age = self.sampler.chain[ok_walkers,:,3].ravel()
            feh = self.sampler.chain[ok_walkers,:,4].ravel()

            if self.fit_for_distance:
                distance = self.sampler.chain[ok_walkers,:,5].ravel()
                AV = self.sampler.chain[ok_walkers,:,6].ravel()
            else:
                distance = None
                AV = 0
            
            lnprob = self.sampler.lnprobability[ok_walkers,:].ravel()

        df = self.ic(mass_A, age, feh, 
                       distance=distance, AV=AV)
        df_B = self.ic(mass_B, age, feh, 
                       distance=distance, AV=AV)
        df_C = self.ic(mass_C, age, feh, 
                       distance=distance, AV=AV)

        for col in df_B.columns:
            m = re.search('_mag$', col)
            if m:
                df['{}_A'.format(col)] = df[col]
                df['{}_B'.format(col)] = df_B[col]
                df['{}_C'.format(col)] = df_C[col]
                df[col] = addmags(df[col], df_B[col], df_C[col])

        df['mass_A'] = df['mass']
        df.drop('mass', axis=1, inplace=True)
        df['mass_B'] = df_B['mass']
        df['mass_C'] = df_C['mass']
        df['radius_B'] = df_B['radius']
        df['radius_C'] = df_C['radius']
        df['Teff_B'] = df_B['Teff']
        df['Teff_C'] = df_C['Teff']
        df['logg_B'] = df_B['logg']
        df['logg_C'] = df_C['logg']
        df['logL_B'] = df_B['logL']
        df['logL_C'] = df_C['logL']
        df['age'] = age
        df['feh'] = feh

        if self.fit_for_distance:
            df['distance'] = distance
            df['AV'] = AV

        df['lnprob'] = lnprob

        self._samples = df.copy()

#class MultipleStarModel(StarModel, BinaryStarModel, TripleStarModel):
#    """
#    StarModel where N_stars (1,2, or 3) is a parameter to estimate.
#    """

#    def binary_loglike(self, *args, **kwargs):
#        return BinaryStarModel.loglike(self, *args, **kwargs)

#    def triple_loglike(self, *args, **kwargs):
#        return TripleStarModel.loglike(self, *args, **kwargs)
        

#### Utility functions #####


def addmags(*mags):
    tot=0
    for mag in mags:
        tot += 10**(-0.4*mag)
    return -2.5*np.log10(tot)
    
def q_prior(q, m=1, gamma=0.3, qmin=0.1):
    """Default prior on mass ratio q ~ q^gamma
    """
    if q < qmin or q > 1:
        return 0
    C = 1/(1/(gamma+1)*(1 - qmin**(gamma+1)))
    return C*q**gamma

def salpeter_prior(m,alpha=-2.35,minmass=0.1,maxmass=10):
    C = (1+alpha)/(maxmass**(1+alpha)-minmass**(1+alpha))
    if m < minmass or m > maxmass:
        return 0
    else:
        return C*m**(alpha)

def local_fehdist(feh):
    """feh PDF based on local SDSS distribution
    
    From Jo Bovy:
    https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
    2D gaussian fit based on Casagrande (2011)
    """
    fehdist= 0.8/0.15*np.exp(-0.5*(feh-0.016)**2./0.15**2.)\
        +0.2/0.22*np.exp(-0.5*(feh+0.15)**2./0.22**2.)

    return fehdist
