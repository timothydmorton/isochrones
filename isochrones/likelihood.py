from isochrones.mags import interp_mag
from numba import jit
from math import pi, log, sqrt
LOG_ONE_OVER_ROOT_2PI = log(1./sqrt(2*pi))


@jit(nopython=True)
def gauss_lnprob(val, unc, model_val):
    resid = val - model_val
    return LOG_ONE_OVER_ROOT_2PI + log(unc) - 0.5 * resid * resid / (unc * unc)


@jit(nopython=True)
def star_lnlike(pars, index_order,
                spec_vals, spec_uncs,
                mag_vals, mag_uncs, i_mags,
                model_grid, i_Teff, i_logg, i_feh, i_Mbol,
                model_ii0, model_ii1, model_ii2,
                bc_grid, bc_ii0, bc_ii1, bc_ii2, bc_ii3):

    Teff, logg, feh, mags = interp_mag(pars, index_order, model_grid,
                                       i_Teff, i_logg, i_feh, i_Mbol,
                                       model_ii0, model_ii1, model_ii2,
                                       bc_grid, i_mags,
                                       bc_ii0, bc_ii1, bc_ii2, bc_ii3)

    lnlike = 0

    # Spec_vals are Teff, logg, feh
    val = spec_vals[0]
    unc = spec_uncs[0]
    if val == val:  # Skip if nan
        lnlike += gauss_lnprob(val, unc, Teff)

    # logg
    val = spec_vals[1]
    unc = spec_uncs[1]
    if val == val:  # Skip if nan
        lnlike += gauss_lnprob(val, unc, logg)

    # feh
    val = spec_vals[2]
    unc = spec_uncs[2]
    if val == val:  # Skip if nan
        lnlike += gauss_lnprob(val, unc, feh)

    for i in range(len(mag_vals)):
        val = mag_vals[i]
        unc = mag_uncs[i]
        lnlike += gauss_lnprob(val, unc, mags[i])

    return lnlike
