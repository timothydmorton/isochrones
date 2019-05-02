import os, sys
import pandas as pd
import numpy as np
import logging

import emcee3
from emcee3.backends import Backend, HDFBackend

class Emcee3Model(emcee3.Model):
    def __init__(self, mod, *args, **kwargs):
        self.mod = mod
        super().__init__(*args, **kwargs)

    def compute_log_prior(self, state):
        state.log_prior = self.mod.lnprior(state.coords)
        return state

    def compute_log_likelihood(self, state):
        state.log_likelihood = self.mod.lnlike(state.coords)
        return state

class Emcee3PriorModel(emcee3.Model):
    def __init__(self, mod, *args, **kwargs):
        self.mod = mod
        super().__init__(*args, **kwargs)

    def compute_log_prior(self, state):
        state.log_prior = self.mod.lnprior(state.coords)
        return state

    def compute_log_likelihood(self, state):
        state.log_likelihood = 0 
        return state

def write_samples(mod, df, resultsdir='results'):
    """df is dataframe of samples, mod is model
    """

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    samplefile = os.path.join(resultsdir, '{}.h5'.format(mod.name))
    df.to_hdf(samplefile, 'samples')

def fit_emcee3(mod, nwalkers=500, verbose=False, nsamples=5000, targetn=4,
                iter_chunksize=200, pool=None, overwrite=False,
                maxiter=10, sample_directory='mcmc_chains',
                nburn=2, mixedmoves=True, resultsdir='mcmc_results',
                prior_only=False, **kwargs):
    """fit model using Emcee3

    modeled after https://github.com/dfm/gaia-kepler/blob/master/fit.py

    nburn is number of autocorr times to discard as burnin.
    """

    # Initialize
    if prior_only:
        walker = Emcee3PriorModel(mod)
    else:
        walker = Emcee3Model(mod)
    ndim = mod.n_params

    if sample_directory is not None:
        sample_file = os.path.join(sample_directory, '{}.h5'.format(mod.name))
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)
        backend = HDFBackend(sample_file)
        try:
            coords_init = backend.current_coords
        except (AttributeError, KeyError):
            coords_init = mod.sample_from_prior(nwalkers, require_valid=True, values=True)
    else:
        backend = Backend()
        coords_init = mod.sample_from_prior(nwalkers, require_valid=True, values=True)

    if mixedmoves:
        moves = [(emcee3.moves.KDEMove(), 0.4),
                 (emcee3.moves.DEMove(1.0), 0.4),
                 (emcee3.moves.DESnookerMove(), 0.2)]
    else:
        moves = emcee3.moves.KDEMove()

    sampler = emcee3.Sampler(moves, backend=backend)
    if overwrite:
        sampler.reset()
        coords_init = mod.sample_from_prior(nwalkers, require_valid=True, values=True)

    if pool is None:
        from emcee3.pools import DefaultPool
        pool = DefaultPool()

    try:
        ensemble = emcee3.Ensemble(walker, coords_init, pool=pool)
    except ValueError:
        import pdb
        pdb.set_trace()

    def calc_stats(s):
        """returns tau_max, neff
        """
        tau = s.get_integrated_autocorr_time(c=1)
        tau_max = tau.max()
        neff = s.backend.niter / tau_max - nburn
        if verbose:
            print("Maximum autocorrelation time: {0}".format(tau_max))
            print("N_eff: {0} ({1})\n".format(neff * nwalkers, neff-nburn))
        return tau_max, neff

    done = False
    if not overwrite:
        try:
            if verbose:
                print('Status from previous run:')
            tau_max, neff = calc_stats(sampler)
            if neff > targetn:
                done = True
        except (emcee3.autocorr.AutocorrError, KeyError):
            pass

    chunksize = iter_chunksize
    for iteration in range(maxiter):
        if done:
            break
        if verbose:
            print("Iteration {0}...".format(iteration + 1))
        sampler.run(ensemble, chunksize, progress=verbose)
        try:
            tau_max, neff = calc_stats(sampler)
        except emcee3.autocorr.AutocorrError:
            tau_max = 0
            continue
        if neff > targetn:
            done = True

    burnin = int(nburn*tau_max)
    ntot = nsamples
    samples = sampler.get_coords(flat=True, discard=burnin)
    total_samples = len(samples)
    if ntot > total_samples:
        ntot = total_samples
    if verbose:
        print("Discarding {0} samples for burn-in".format(burnin))
        print("Randomly choosing {0} samples".format(ntot))
    inds = np.random.choice(total_samples, size=ntot, replace=False)
    samples = samples[inds]

    df = pd.DataFrame(samples, columns=mod.param_names)
    write_samples(mod, df, resultsdir=resultsdir)
    
    return df
    # return sampler    
