import numpy as np
import numba as nb
from math import log10

from .interp import interp_value_3d, interp_value_4d


@nb.jit(nopython=True)
def interp_mag(pars, index_order, model_grid,
               i_Teff, i_logg, i_feh, i_Mbol,
               model_ii0, model_ii1, model_ii2,
               bc_grid, bc_cols,
               bc_ii0, bc_ii1, bc_ii2, bc_ii3):
    """

    pars: 2d array
    """
    # logTeff, logg, logL returned.
    ipar0 = index_order[0]
    ipar1 = index_order[1]
    ipar2 = index_order[2]
    star_props = interp_value_3d(pars[ipar0], pars[ipar1], pars[ipar2],
                                 model_grid, [i_Teff, i_logg, i_feh, i_Mbol],
                                 model_ii0, model_ii1, model_ii2)
    Teff = star_props[0]
    logg = star_props[1]
    feh = star_props[2]
    ipar4 = index_order[4]
    AV = pars[ipar4]
    bc = interp_value_4d(Teff, logg, feh, AV,
                         bc_grid, bc_cols,
                         bc_ii0, bc_ii1, bc_ii2, bc_ii3)

    mBol = star_props[3]
    ipar3 = index_order[3]
    dist_mod = 5 * log10(pars[ipar3]/10.)

    n_bands = len(bc_cols)
    mags = np.empty(n_bands, dtype=nb.float64)
    for i in range(n_bands):
        mags[i] = mBol + dist_mod - bc[i]

    return Teff, logg, feh, mags


@nb.jit(nopython=True)
def interp_mags(pars, index_order, model_grid,
                i_Teff, i_logg, i_feh, i_Mbol,
                model_ii0, model_ii1, model_ii2,
                bc_grid, bc_cols,
                bc_ii0, bc_ii1, bc_ii2, bc_ii3):
    """
    pars is n_values x 5
    """
    n_pars = pars.shape[0]
    n_values = pars.shape[1]
    n_bands = len(bc_cols)

    Teffs = np.empty(n_values, dtype=nb.float64)
    loggs = np.empty(n_values, dtype=nb.float64)
    fehs = np.empty(n_values, dtype=nb.float64)
    mags = np.empty((n_values, n_bands), dtype=nb.float64)

    p = np.empty(n_pars)
    for i in range(n_values):
        for j in range(n_pars):
            p[j] = pars[j, i]

        Teff, logg, feh, mag = interp_mag(p, index_order, model_grid,
                                          i_Teff, i_logg, i_feh, i_Mbol,
                                          model_ii0, model_ii1, model_ii2,
                                          bc_grid, bc_cols,
                                          bc_ii0, bc_ii1, bc_ii2, bc_ii3)
        Teffs[i] = Teff
        loggs[i] = logg
        fehs[i] = feh
        for j in range(n_bands):
            mags[i, j] = mag[j]

    return Teffs, loggs, fehs, mags
