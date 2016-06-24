from __future__ import print_function, division

import numpy as np
import pandas as pd
import os, os.path, sys, re, glob

import numpy.random as rand
import logging
import json
#import emcee
import corner
#import pymultinest

from configobj import ConfigObj

from .utils import addmags
from .observation import ObservationTree, Observation, Source 
from .priors import age_prior, distance_prior, AV_prior, q_prior
from .priors import salpeter_prior, local_fehdist

class StarModel(object):
    """

    :param ic: 
        :class:`Isochrone` object used to model star.

    :param obs: (optional)
        :class:`ObservationTree` object containing photometry information.
        If not provided, then one will be constructed from the provided
        keyword arguments (which must include at least one photometric
            bandpass).  This should only happen in the simplest case
        of a single star system---if multiple stars are detected
        in any of the observations being used, an :class:`ObservationTree`
        should be passed.

    :param N:
        Number of model stars to assign to each "leaf node" of the 
        :class:`ObservationTree`.    

    :param **kwargs:
            Keyword arguments must be properties of given isochrone, e.g., logg,
            feh, Teff, and/or magnitudes.  The values represent measurements of
            the star, and must be in (value,error) format. All such keyword
            arguments will be held in ``self.properties``.  ``parallax`` is
            also a valid property, and should be provided in miliarcseconds.        
    """
    def __init__(self, ic, obs=None, N=1, index=0,
                 name='', use_emcee=False,
                 **kwargs):

        self.name = name
        self._ic = ic

        self.use_emcee = use_emcee

        # If obs is not provided, build it
        if obs is None:
            self._build_obs(**kwargs)
            self.obs.define_models(ic, N=N, index=index)
            self._add_properties(**kwargs)
        else:
            self.obs = obs


        self._priors = {'mass':salpeter_prior,
                        'feh':local_fehdist,
                        'q':q_prior,
                        'age':age_prior,
                        'distance':distance_prior,
                        'AV':AV_prior}
        self._bounds = {'mass':None,
                        'feh':None,
                        'age':None,
                        'q':(0.1,1.0),
                        'distance':(0,3000.),
                        'AV':(0,1.)}

        self._samples = None

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    @classmethod
    def from_ini(cls, ic, folder='.', ini_file='star.ini'):
        """
        Initialize a StarModel from a .ini file

        The "classic" format (version <= 0.9) should still work for a single star,
        where all properties are just listed in the file; e.g.,

            J = 10, 0.05
            H = 9.5, 0.05
            K = 9.0, 0.05
            Teff = 5000, 150

        If multiple stars are observed, please use the `obsfile` keyword,
        pointing to a file with the summarized photometric observations.
        In this case, spectroscopic/parallax info should still be included
        in the .ini file; e.g.,

            obsfile = obs.csv
            Teff = 5000, 150

        The obsfile should be a comma-separated table with the following columns:
        `[name, band, resolution, mag, e_mag, separation, pa, relative]`.

          * `name` is the name of instrument
          * `band` is the photometric bandpass
          * `resolution` is the approximate spatial resolution of instrument
          * `mag`, `e_mag` describe magnitude of source (absolute or relative)
          * `separation`, `pa` describe position of source
          * `relative`: single-bit flag; if 1 then magnitudes taken with this
            instrument are assumed to be relative rather than absolute.

        If `obsfile` is provided as above, then also the `N` and `index`
        parameters may be provided, to specify the relations between the 
        model stars.  If these are not provided, then `N` will default to `1`
        (one model star per star observed in highest-resolution observation)
        and `index` will default to all `0` (all stars physically associated).

        """
        if not os.path.isabs(ini_file):
            ini_file = os.path.join(folder,ini_file)

        if type(ic) == type(type):
            ic = ic()

        logging.debug('Initializing StarModel from {}'.format(ini_file))

        config = ConfigObj(ini_file)

        if 'N' in config:
            try:
                N = int(config['N'])
            except:
                N = [int(n) for n in config['N']]
        else:
            N = 1

        if 'index' in config:
            try:
                index = int(config['index'])
            except:
                try:
                    index = [int(i) for i in config['index']]
                except:
                    index = [[int(i) for i in inds] for inds in config['index']]
        else:
            index = 0

        # Make, e.g., N=2, index=[0,1] work 
        if type(index)==list:
            if len(index)==N:
                index = [index]

        logging.debug('N={}, index={}'.format(N, index))

        if 'obsfile' in config:
            obsfile = config['obsfile']
            if not os.path.isabs(obsfile):
                obsfile = os.path.join(folder, obsfile)

            df = pd.read_csv(obsfile)
            obs = ObservationTree.from_df(df)
            obs.define_models(ic, N=N, index=index)
            for prop in ['Teff','logg','feh']:
                if prop in config:
                    val = [float(v) for v in config[prop]]
                    obs.add_spectroscopy(**{prop:val})
            if 'parallax' in config:
                val = [float(v) for v in config['parallax']]
                obs.add_parallax(val)
                
            new = cls(ic, obs=obs, N=N, index=index)

        else:
            kwargs = {}
            for kw in config.keys():
                if kw in ic.bands or kw in ['Teff','logg','feh','parallax']:
                    try:
                        kwargs[kw] = float(config[kw])
                    except:
                        kwargs[kw] = (float(config[kw][0]), float(config[kw][1]))

            new = cls(ic, N=N, index=index, **kwargs)

        return new
            
        

    def bounds(self, prop):
        if self._bounds[prop] is not None:
            return self._bounds[prop]
        elif prop=='mass':
            self._bounds['mass'] = (self.ic.minmass,
                                    self.ic.maxmass)
        elif prop=='feh':
            self._bounds['feh'] = (self.ic.minfeh,
                                   self.ic.maxfeh)
        elif prop=='age':
            self._bounds['age'] = (self.ic.minage,
                                   self.ic.maxage)
        else:
            raise ValueError('Unknown property {}'.format(prop))
        return self._bounds[prop]

    def set_bounds(self, prop, val):
        if len(val)!=2:
            raise ValueError('Must provide (min, max)')
        self._bounds[prop] = val

    def _build_obs(self, **kwargs):
        """
        Builds ObservationTree out of keyword arguments

        Ignores anything that is not a photometric bandpass.
        This should not be used if there are multiple stars observed.

        Creates self.obs
        """
        tree = ObservationTree()
        for k,v in kwargs.items():
            if k in self.ic.bands:
                if np.size(v) != 2:
                    logging.warning('{}={} ignored.'.format(k,v))
                    continue
                o = Observation('',k,99) #bogus resolution=99
                o.add_source(Source(v[0],v[1]))
                tree.add_observation(o)
        self.obs = tree

    def _add_properties(self, **kwargs):
        """
        Adds non-photometry properties to ObservationTree
        """
        for k,v in kwargs.items():
            if k=='parallax':
                self.obs.add_parallax(v)
            elif k not in self.ic.bands:
                par = {k:v}
                self.obs.add_spectroscopy(**par)

    def lnpost(self, p):
        lnpr = self.lnprior(p)
        if not np.isfinite(lnpr):
            return lnpr
        return lnpr + self.lnlike(p)

    def lnlike(self, p, **kwargs):
        lnl = self.obs.lnlike(p, **kwargs)
        return lnl

    def lnprior(self, p):
        N = self.obs.Nstars
        i = 0
        lnp = 0
        for s in self.obs.systems:
            age, feh, dist, AV = p[i+N[s]:i+N[s]+4]
            for prop, val in zip(['age','feh','distance','AV'],
                                 [age, feh, dist, AV]):
                lo,hi = self.bounds(prop)
                if val < lo or val > hi:
                    return -np.inf
                lnp += np.log(self.prior(prop, val, 
                                  bounds=self.bounds(prop)))
                if not np.isfinite(lnp):
                    logging.debug('lnp=-inf for {}={} (system {})'.format(prop,val,s))
                    return -np.inf

            # Note: this is just assuming proper order.
            #  Is this OK?  Should keep eye out for bugs here.

            masses = p[i:i+N[s]]

            # Mass prior for primary
            lnp += np.log(self.prior('mass', masses[0],
                                bounds=self.bounds('mass')))
            if not np.isfinite(lnp):
                logging.debug('lnp=-inf for mass={} (system {})'.format(masses[0],s))

            # Priors for mass ratios
            for j in range(N[s]-1):
                q = masses[j+1]/masses[0]
                qmin, qmax = self.bounds('q')

                ## The following would enforce MA > MB > MC, but seems to make things very slow:
                #if j+1 > 1:
                #    qmax = masses[j] / masses[0]

                lnp += np.log(self.prior('q', q,
                                         bounds=(qmin,qmax)))
                if not np.isfinite(lnp):
                    logging.debug('lnp=-inf for q={} (system {})'.format(val,s))
                    return -np.inf

            i += N[s] + 4

        return lnp

    def prior(self, prop, val, **kwargs):
        return self._priors[prop](val, **kwargs)
    

    @property
    def n_params(self):
        tot = 0
        for _,n in self.obs.Nstars.items():
            tot += 4+n
        return tot

    def mnest_prior(self, cube, ndim, nparams):
        i = 0
        for _,n in self.obs.Nstars.items():
            minmass, maxmass = self.bounds('mass')
            for j in xrange(n):
                cube[i+j] = (maxmass - minmass)*cube[i+j] + minmass
            
            for j, par in enumerate(['age','feh','distance','AV']):
                lo, hi = self.bounds(par)
                cube[i+n+j] = (hi - lo)*cube[i+n+j] + lo
            i += 4 + n

    def mnest_loglike(self, cube, ndim, nparams):
        """loglikelihood function for multinest
        """
        return self.lnpost(cube)
            
    @property
    def labelstring(self):
        s = ''
        for l in self.obs.leaf_labels:
            s += l+'-'
        return s[:-1]

    def fit_multinest(self, n_live_points=1000, basename=None,
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

        if basename is None:
            s = self.labelstring
            if s=='0_0':
                s = 'single'
            elif s=='0_0-0_1':
                s = 'binary'
            elif s=='0_0-0_1-0_2':
                s = 'triple'
            #s += '-'
            basename = os.path.join('chains',s) 

        folder = os.path.abspath(os.path.dirname(basename))
        if not os.path.exists(folder):
            os.makedirs(folder)

        #If previous fit exists, see if it's using the same
        # observed properties
        prop_nomatch = False
        propfile = '{}properties.json'.format(basename)
        
        """
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
        """

        if refit or overwrite:
            files = glob.glob('{}*'.format(basename))
            [os.remove(f) for f in files]

        self._mnest_basename = basename

        pymultinest.run(self.mnest_loglike, self.mnest_prior, self.n_params,
                        n_live_points=n_live_points, outputfiles_basename=basename,
                        verbose=verbose,
                        **kwargs)

        #with open(propfile, 'w') as f:
        #    json.dump(self.properties, f, indent=2)

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



    def emcee_p0(self, nwalkers):
        p0 = []
        for _,n in self.obs.Nstars.items():
            m0, age0, feh0 = self.ic.random_points(nwalkers)
            _, max_distance = self.bounds('distance')
            _, max_AV = self.bounds('AV')
            d0 = 10**(rand.uniform(0,np.log10(max_distance),size=nwalkers))
            AV0 = rand.uniform(0, max_AV, size=nwalkers)

            # This will occasionally give masses outside range.
            for i in range(n):
                p0 += [m0 * 0.95**i]
            p0 += [age0, feh0, d0, AV0]
        return np.array(p0).T

    def fit_mcmc(self,nwalkers=300,nburn=200,niter=100,
                 p0=None,initial_burn=None,
                 ninitial=50, loglike_kwargs=None,
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
            
        npars = self.n_params

        if p0 is None:
            p0 = self.emcee_p0(nwalkers)
            if initial_burn:
                sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost,
                                                **kwargs)
                #ninitial = 300 #should this be parameter?
                pos, prob, state = sampler.run_mcmc(p0, ninitial) 

                # Choose walker with highest final lnprob to seed new one
                i,j = np.unravel_index(sampler.lnprobability.argmax(),
                                        sampler.shape)
                p0_best = sampler.chain[i,j,:]
                print("After initial burn, p0={}".format(p0_best))
                p0 = p0_best * (1 + rand.normal(size=p0.shape)*0.001)
                print(p0)
        else:
            p0 = np.array(p0)
            p0 = rand.normal(size=(nwalkers,npars))*0.01 + p0.T[None,:]
        
        sampler = emcee.EnsembleSampler(nwalkers,npars,self.lnpost)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0=state)
        
        self._sampler = sampler
        return sampler

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
            lnprob = chain[:,-1]
            chain = chain[:,:-1]
        else:
            #select out only walkers with > 0.15 acceptance fraction
            ok = self.sampler.acceptance_fraction > 0.15

            chain = self.sampler.chain[ok,:,:]
            chain = chain.reshape((chain.shape[0]*chain.shape[1],
                                        chain.shape[2]))

            lnprob = self.sampler.lnprobability[ok, :].ravel()

        df = pd.DataFrame()

        i=0
        for s,n in self.obs.Nstars.items():
            age = chain[:,i+n]
            feh = chain[:,i+n+1]
            distance = chain[:,i+n+2]
            AV = chain[:,i+n+3]
            for j in range(n):
                mass = chain[:,i+j]
                d = self.ic(mass, age, feh, 
                             distance=distance, AV=AV)
                for c in d.columns:
                    df[c+'_{}_{}'.format(s,j)] = d[c]
            df['age_{}'.format(s)] = age
            df['feh_{}'.format(s)] = feh
            df['distance_{}'.format(s)] = distance
            df['AV_{}'.format(s)] = AV
            
            i += 4 + n

        for b in self.ic.bands:
            tot = np.inf
            for s,n in self.obs.Nstars.items():
                for j in range(n):
                    tot = addmags(tot,df[b + '_mag_{}_{}'.format(s,j)])
            df[b + '_mag'] = tot

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
        
    def triangle(self, *args, **kwargs):
        return self.corner(*args, **kwargs)

    def corner(self, params, query=None, **kwargs):
        df = self.samples
        if query is not None:
            df = df.query(query)

        fig = corner.corner(df[params], labels=params, **kwargs)
        fig.suptitle(self.name, fontsize=22)
        return fig

    def triangle_physical(self, *args, **kwargs):
        return self.corner_physical(*args, **kwargs)

    def corner_physical(self, props=['mass','radius','feh','distance'], **kwargs):
        collective_props = ['feh','age','distance','AV']
        indiv_props = [p for p in props if p not in collective_props]
        sys_props = [p for p in props if p in collective_props]
        
        props = ['{}_{}'.format(p,l) for p in indiv_props for l in self.obs.leaf_labels]
        props += ['{}_{}'.format(p,s) for p in sys_props for s in self.obs.systems]
        
        return self.corner(props, **kwargs)


    def save_hdf(self, filename, path='', overwrite=False, append=False):
        """Saves object data to HDF file (only works if MCMC is run)

        Samples are saved to /samples location under given path,
        :class:`ObservationTree` is saved to /obs location under given path. 
        
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

        if self.samples is not None:
            self.samples.to_hdf(filename, path+'/samples')
        else:
            pd.DataFrame().to_hdf(filename, path+'/samples')

        self.obs.save_hdf(filename, path+'/obs', append=True)
        
        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/samples'.format(path)).attrs
        
        attrs.ic_type = type(self.ic)
        attrs.use_emcee = self.use_emcee
        attrs._mnest_basename = self._mnest_basename
        
        attrs._bounds = self._bounds
        attrs._priors = self._priors

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
            samples = store[path+'/samples']
            attrs = store.get_storer(path+'/samples').attrs        
        except:
            store.close()
            raise
        
        ic = attrs.ic_type
        use_emcee = attrs.use_emcee
        basename = attrs._mnest_basename
        bounds = attrs._bounds
        priors = attrs._priors

        if name is None:
            try:
                name = attrs.name
            except:
                name = ''

        store.close()

        obs = ObservationTree.load_hdf(filename, path+'/obs')
        
        mod = cls(ic, obs=obs, 
                  use_emcee=use_emcee, name=name)
        mod._samples = samples
        mod._mnest_basename = basename
        return mod
