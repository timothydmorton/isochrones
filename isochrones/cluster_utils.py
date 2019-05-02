from math import log10, log, exp
import numba as nb
import numpy as np

from .utils import trapz
from .priors import powerlaw_lnpdf


@nb.jit(nopython=True)
def logaddexp(x1, x2):
    xmax = max(x1, x2)
    return xmax + log(exp(x1 - xmax) + exp(x2 - xmax))


@nb.jit(nopython=True)
def logsumexp(xx):
    xmax = xx[0]
    n = len(xx)
    for i in range(n):
        if xx[i] > xmax:
            xmax = xx[i]

    expsum = 0
    for i in range(n):
        expsum += exp(xx[i] - xmax)

    return xmax + log(expsum)


@nb.jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def calc_lnlike_grid(lnlike_prop,
                     model_mags, Nbands,
                     masses, ln_dm_deeps, eeps,
                     mag_values, mag_uncs,
                     alpha, gamma, fB,
                     mass_lo, mass_hi, q_lo):
    """Returns half-filled NxN array of lnlike(phot) + lnlike(mass), as function of E1, E2

    Lots of different shaped arrays here.

    lnlike_prop: (Nstars, Neep)
    model_mags: (Neep, Nbands)
    Nbands: int
    masses: Neep
    dm_deeps: Neep
    eeps: Neep
    mag_values: (Nstars, Nbands)
    mag_uncs: (Nstars, Nbands)
    alpha, gamma, fB, mass_lo, mass_hi, q_lo: float

    """
    n = len(model_mags)
    n_stars = len(mag_values)

    lnlikes = np.zeros((n_stars, n, n))

    for i in nb.prange(n_stars):
        for j in range(n):
            for k in range(j+1):
                if masses[k] / masses[j] < q_lo:
                    lnlikes[i, j, k] = -np.inf
                else:
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

                        # like_phot = fB * exp(lnlike_phot_binary) + (1 - fB) * exp(lnlike_phot_single)
                        # lnlike_phot += log(like_phot)

                        lnlike_phot += logaddexp(log(fB) + lnlike_phot_binary,
                                                 log(1 - fB) + lnlike_phot_single)

                    # ln(likelihood) for mass
                    # lnlike_mass = powerlaw_lnpdf(masses[j] + masses[k], alpha, mass_lo, mass_hi)
                    lnlike_mass = powerlaw_lnpdf(masses[j], alpha, mass_lo, mass_hi)
                    lnlike_mass += ln_dm_deeps[j]

                    # ln(likelihood) for mass ratio
                    lnlike_mass_ratio = powerlaw_lnpdf(masses[k] / masses[j], gamma, q_lo, 1.)

                    lnlikes[i, j, k] = lnlike_phot + lnlike_mass + lnlike_mass_ratio + lnlike_prop[i, j]

    return lnlikes


@nb.jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def integrate_over_eeps(lnlike_grid, eeps, Nstars):

    likes_marginalized = np.zeros(Nstars)
    n = len(eeps)
    for i in nb.prange(Nstars):
        row = np.zeros(n)
        for j in range(n):
            tot = 0
            m = j + 1
            for k in range(m - 1):
                k2 = k + 1
                tot += 0.5 * (exp(lnlike_grid[i, j, k]) + exp(lnlike_grid[i, j, k2])) * (eeps[k2] - eeps[k])

            row[j] = tot  # * n / (m - 1) # should I rescale for equal weights per column?
            # if tot > 0:
            #     print(i, eeps[j], tot)

        likes_marginalized[i] = trapz(row, eeps)

    return likes_marginalized
