from __future__ import print_function, division

import numpy as np
import pandas as pd
import numpy.random as rand
import logging
import emcee

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

    :param maxAV: (optional)
        Maximum allowed extinction (i.e. the extinction @ infinity in direction of star).  Default is 1.

    :param max_distance: (optional)
        Maximum allowed distance (pc).  Default is 3000.

    :param **kwargs:
            Keyword arguments must be properties of given isochrone, e.g., logg,
            feh, Teff, and/or magnitudes.  The values represent measurements of
            the star, and must be in (value,error) format. All such keyword
            arguments will be held in ``self.properties``.  ``parallax`` is
            also a valid property, and should be provided in miliarcseconds.        
    """
    def __init__(self, ic, obs=None, N=1, index=0,
                 maxAV=1., max_distance=3000.,
                 min_logg=None, name='', **kwargs):

        self.maxAV = maxAV
        self.max_distance = max_distance
        self.min_logg = None
        self.name = name
        self._ic = ic

        # If obs is not provided, build it
        if obs is None:
            self._build_obs(**kwargs)
        else:
            self.obs = obs

        self.obs.define_models(ic, N=N, index=index)
        self._add_properties(**kwargs)

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
                        'distance':(0,self.max_distance),
                        'AV':(0,self.maxAV)}

        self._samples = None

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

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
                lnp += np.log(self.prior('q', q,
                            bounds=self.bounds('q')))
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

        if False:#not self.use_emcee:
            chain = np.loadtxt('{}post_equal_weights.dat'.format(self._mnest_basename))

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
