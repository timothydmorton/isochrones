from math import log10, pow, log, exp
from numba import jit, prange
import numpy as np

from .utils import trapz
from .priors import powerlaw_lnpdf

@jit(nopython=True, parallel=True, nogil=True)
def calc_lnlike_grid(lnlike_prop,
                     model_mags, Nbands,
                     masses, eeps,
                     mag_values, mag_uncs,
                     alpha, gamma, fB,
                     mass_lo, mass_hi, q_lo):
    """Returns half-filled NxN array of lnlike(phot) + lnlike(mass), as function of E1, E2

    Lots of different shaped arrays here.

    lnlike_prop: (Nstars, Neep)
    model_mags: (Neep, Nbands)
    Nbands: int
    masses: Neep
    eeps: Neep
    mag_values: (Nstars, Nbands)
    mag_uncs: (Nstars, Nbands)
    alpha, gamma, fB, mass_lo, mass_hi, q_lo: float

    """
    n = len(model_mags)
    n_stars = len(mag_values)

    lnlikes = np.zeros((n_stars, n, n))

    for i in prange(n_stars):
        for j in range(n):
            for k in range(j+1):
                lnlike_phot = 0
                for b in range(Nbands):
                    # ln(likelihood) for photometry for binary model
                    f1 = 10**(-0.4 * model_mags[j, b])
                    f2 = 10**(-0.4 * model_mags[k, b])
                    mag_value = mag_values[i, b]
                    mag_unc = mag_uncs[i, b]
                    tot_mag_binary = -2.5 * log10(f1 + f2)
                    resid_binary = tot_mag_binary - mag_value
                    lnlike_phot_binary = -0.5 * resid_binary * resid_binary / (mag_unc * mag_unc)

                    resid_single = model_mags[j, b] - mag_value
                    lnlike_phot_single = -0.5 * resid_single * resid_single / (mag_unc * mag_unc)

                    like_phot = fB * exp(lnlike_phot_binary) + (1 - fB) * exp(lnlike_phot_single)
                    lnlike_phot += log(like_phot)

                # ln(likelihood) for total mass
                lnlike_mass = powerlaw_lnpdf(masses[j] + masses[k], alpha, mass_lo, mass_hi)

                # ln(likelihood) for mass ratio
                lnlike_mass_ratio = powerlaw_lnpdf(masses[k] / masses[j], gamma, q_lo, 1.)

                lnlikes[i, j, k] = lnlike_phot + lnlike_mass + lnlike_mass_ratio + lnlike_prop[i, j]

    return lnlikes


@jit(nopython=True, parallel=True, nogil=True)
def integrate_over_eeps(lnlike_grid, eeps, Nstars):

    likes_marginalized  = np.zeros(Nstars)
    n = len(eeps)
    for i in prange(Nstars):
        row = np.zeros(n)
        for j in range(n):
            tot = 0
            m = j + 1
            for k in range(m - 1):
                k2 = k + 1
                tot += 0.5 * (exp(lnlike_grid[i, j, k]) + exp(lnlike_grid[i, j, k2])) * (eeps[k2] - eeps[k])

            row[j] = tot #* n / (m - 1) # rescale for equal weights per column.  Is this right?
            # if tot > 0:
            #     print(i, eeps[j], tot)

        likes_marginalized[i] = trapz(row, eeps)

    return likes_marginalized
