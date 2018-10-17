import re

import numpy as np
import pandas as pd

from isochrones import StarModel
from isochrones.priors import PowerLawPrior, FlatLogPrior, FehPrior, FlatPrior

class StarClusterModel(object):

    param_names = ['age', 'feh', 'AV', 'distance', 'gamma']

    def __init__(self, ic, stars,
                 halo_fraction=0.001, max_AV=1., max_distance=50000):
        self.ic = ic
        self.stars = stars
        self.bands = [c for c in stars.columns if not re.search('unc', c)]

        self.priors = {'age': FlatLogPrior((6, 10.15)),
                       'feh': FehPrior(halo_fraction=halo_fraction),
                       'AV' : FlatPrior((0, max_AV)),
                       'distance' : PowerLawPrior(alpha=2., bounds=(0, max_distance)),
                       'gamma' : FlatPrior((-5, 0))}

    def lnprior(self, p):
        age, feh, distance, AV, gamma = p

        lnp = 0
        for prop in ['age', 'feh', 'distance', 'AV', 'gamma']:
            val = np.log(self.priors[prop](eval(prop)))
            if not np.isfinite(val):
                print(prop, val)
            lnp += val

        if not np.isfinite(lnp):
            return -np.inf

        return lnp

    def lnlike(self, p):
        age, feh, distance, AV, gamma = p

        lnlike_tot = 0
        eeps = self.ic.eeps


        # Compute log-likelihood of each mass under power-law distribution
        #  Also use this opportunity to find the valid range of EEP
        mass_fn = PowerLawPrior(gamma, bounds=(self.ic.minmass, self.ic.maxmass))
        model_masses = self.ic.initial_mass(eeps, age, feh)
        ok = np.isfinite(model_masses)

        model_masses = model_masses[ok]
        eeps = eeps[ok]

        lnlike_mass = np.log(mass_fn.pdf(model_masses))

        # Compute log-likelihood of observed photometry
        model_mags = {b : self.ic.mag[b](eeps, age, feh, distance, AV) for b in self.bands}

        lnlike_phot = 0
        for b in self.bands:
            vals = self.stars[b].values
            uncs = self.stars[b + '_unc'].values

            lnlike_phot += -0.5 * (vals - model_mags[b][:, None])**2 / uncs**2

        integrand = np.exp(lnlike_mass[:, None] + lnlike_phot)

        like_tot = np.trapz(integrand, axis=1)

        ok = (like_tot != 0)
        return np.log(like_tot[ok]).sum()

    def lnpost(self, p):
        return self.lnprior(p) + self.lnlike(p)
