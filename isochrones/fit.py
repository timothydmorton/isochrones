import os, sys
import pandas as pd
import numpy as np

import emcee3
from emcee3.backends import Backend, HDFBackend

class Emcee3Model(emcee3.Model):
    def __init__(self, mod, *args, **kwargs):
        self.mod = mod
        super(Emcee3Model, self).__init__(*args, **kwargs)

    def compute_log_prior(self, state):
        state.log_prior = self.mod.lnprior(state.coords)
        return state

    def compute_log_likelihood(self, state):
        state.log_likelihood = self.mod.lnlike(state.coords)
        return state

def fit_emcee3(mod, nwalkers=500, verbose=False, nsamples=10000, targetn=6,
                iter_chunksize=100, pool=None, overwrite=False,
                maxiter=100, sample_directory='mcmc_chains',
                nburn=3, mixedmoves=True, **kwargs):
    """fit model using Emcee3 

    modeled after https://github.com/dfm/gaia-kepler/blob/master/fit.py

    nburn is number of autocorr times to discard as burnin.
    """

    # Initialize
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
            coords_init = mod.sample_from_prior(nwalkers)
    else:
        backend = Backend()
        coords_init = mod.sample_from_prior(nwalkers)

    if mixedmoves:
        moves = [(emcee3.moves.KDEMove(), 0.4),
                 (emcee3.moves.DEMove(1.0), 0.4),
                 (emcee3.moves.DESnookerMove(), 0.2)]
    else:
        moves = emcee3.moves.KDEMove()

    sampler = emcee3.Sampler(moves, backend=backend)
    if overwrite:
        sampler.reset()
        coords_init = mod.sample_from_prior(nwalkers)

    if pool is None:
        from emcee3.pools import DefaultPool
        pool = DefaultPool()

    ensemble = emcee3.Ensemble(walker, coords_init, pool=pool)

    def calc_stats(s):
        """returns tau_max, neff
        """
        tau = s.get_integrated_autocorr_time(c=1)
        tau_max = tau.max()
        neff = s.backend.niter / tau_max - nburn
        if verbose:
            print("Maximum autocorrelation time: {0}".format(tau_max))
            print("N_eff: {0}\n".format(neff * nwalkers))            
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
            continue
        if neff > targetn:
            done = True

    burnin = int(nburn*tau_max)
    ntot = nsamples
    if verbose:
        print("Discarding {0} samples for burn-in".format(burnin))
        print("Randomly choosing {0} samples".format(ntot))
    samples = sampler.get_coords(flat=True, discard=burnin)
    total_samples = len(samples)
    inds = np.random.choice(np.arange(len(samples)), size=ntot, replace=False)
    samples = samples[inds]

    df = pd.DataFrame(samples, columns=mod.param_names)
    
    return df