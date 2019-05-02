from isochrones.mags import interp_mag
from .utils import fast_addmags
import numba as nb
from math import pi, log, sqrt


LOG_ONE_OVER_ROOT_2PI = log(1./sqrt(2*pi))


@nb.jit(nopython=True)
def gauss_lnprob(val, unc, model_val):
    resid = val - model_val
    return LOG_ONE_OVER_ROOT_2PI + log(unc) - 0.5 * resid * resid / (unc * unc)


@nb.jit(nopython=True)
def star_lnlike(pars, index_order,
                spec_vals, spec_uncs,
                mag_vals, mag_uncs, i_mags,
                model_grid, i_Teff, i_logg, i_feh, i_Mbol,
                model_ii0, model_ii1, model_ii2,
                bc_grid, bc_ii0, bc_ii1, bc_ii2, bc_ii3):

    n_pars = len(pars)
    has_binary = False
    has_triple = False
    if n_pars == 5:
        single_pars = [pars[0], pars[1], pars[2], pars[3], pars[4]]
    elif n_pars == 6:  # binary system
        single_pars = [pars[0], pars[2], pars[3], pars[4], pars[5]]
        binary_pars = [pars[1], pars[2], pars[3], pars[4], pars[5]]
        has_binary = True
    elif n_pars == 7:  # triple system
        single_pars = [pars[0], pars[3], pars[4], pars[5], pars[6]]
        binary_pars = [pars[1], pars[3], pars[4], pars[5], pars[6]]
        triple_pars = [pars[2], pars[3], pars[4], pars[5], pars[6]]
        has_binary = True
        has_triple = True

    Teff, logg, feh, mags = interp_mag(single_pars, index_order, model_grid,
                                       i_Teff, i_logg, i_feh, i_Mbol,
                                       model_ii0, model_ii1, model_ii2,
                                       bc_grid, i_mags,
                                       bc_ii0, bc_ii1, bc_ii2, bc_ii3)

    if has_binary:
        _, _, _, mags_binary = interp_mag(binary_pars, index_order, model_grid,
                                          i_Teff, i_logg, i_feh, i_Mbol,
                                          model_ii0, model_ii1, model_ii2,
                                          bc_grid, i_mags,
                                          bc_ii0, bc_ii1, bc_ii2, bc_ii3)

    if has_triple:
        _, _, _, mags_triple = interp_mag(triple_pars, index_order, model_grid,
                                          i_Teff, i_logg, i_feh, i_Mbol,
                                          model_ii0, model_ii1, model_ii2,
                                          bc_grid, i_mags,
                                          bc_ii0, bc_ii1, bc_ii2, bc_ii3)

    if n_pars == 6:
        for i in range(len(mags)):
            mags[i] = fast_addmags([mags[i], mags_binary[i]])
    elif n_pars == 7:
        for i in range(len(mags)):
            mags[i] = fast_addmags([mags[i], mags_binary[i], mags_triple[i]])

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
