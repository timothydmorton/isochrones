from numba import jit, float64
from math import sqrt
import numpy as np

@jit(nopython=True)
def interp_box(x, y, z, box, values, p=-2):
    """
    box is 8x3 array, though not really a box, necessarily
    
    values is length-8 array, corresponding to values at the "box" coords

    """
    
    # Calculate the distance to each vertex
    
    val = 0
    norm = 0
    for i in range(8):
        # Inv distance, or Inv-dsq weighting
        dsq = (x-box[i,0])**2 + (y-box[i,1])**2 + (z-box[i, 2])**2
        if dsq==0:
            return values[i]
        else:
            w = dsq**(p/2.)
            val += w * values[i]
            norm += w
    
    return val/norm

@jit(nopython=True)
def searchsorted(arr, N, x):
    """N is length of arr
    """
    L = 0
    R = N-1
    done = False
    m = (L+R)//2
    while not done:
        if arr[m] < x:
            L = m + 1
        elif arr[m] > x:
            R = m - 1
        elif arr[m] == x:
            done = True
        m = (L+R)//2
        if L>R:
            done = True
    return L
        
@jit(nopython=True)
def searchsorted_many(arr, values):
    N = len(arr)
    Nval = len(values)
    inds = np.zeros(Nval)
    for i in range(Nval):
        x = values[i]
        L = 0
        R = N-1
        done = False
        m = (L+R)//2
        while not done:
            if arr[m] < x:
                L = m + 1
            elif arr[m] > x:
                R = m - 1
            m = (L+R)//2
            if L>R:
                done = True
        inds[i] = L
    return inds

@jit(nopython=True)
def interp_values(mass_arr, age_arr, feh_arr, icol, 
                 grid, mass_col, ages, fehs, grid_Ns):
    """mass_arr, age_arr, feh_arr are all arrays at which values are desired

    icol is the column index of desired value
    grid is nfeh x nage x max(nmass) x ncols array
    mass_col is the column index of mass
    ages is grid of ages
    fehs is grid of fehs
    grid_Ns keeps track of nmass in each slice (beyond this are nans)
    
    """
    
    N = len(mass_arr)
    results = np.zeros(N)

    Nage = len(ages)
    Nfeh = len(fehs)

    for i in range(N):
        results[i] = interp_value(mass_arr[i], age_arr[i], feh_arr[i], icol, 
                                 grid, mass_col, ages, fehs, grid_Ns, False)

        ## Things are slightly faster if the below is used, but for consistency,
        ## using above.
        # mass = mass_arr[i]
        # age = age_arr[i]
        # feh = feh_arr[i]

        # ifeh = searchsorted(fehs, Nfeh, feh)
        # iage = searchsorted(ages, Nage, age)
        # if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
        #     results[i] = np.nan
        #     continue

        # pts = np.zeros((8,3))
        # vals = np.zeros(8)

        # i_f = ifeh - 1
        # i_a = iage - 1
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[0, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[0, 1] = ages[i_a]
        # pts[0, 2] = fehs[i_f]
        # vals[0] = grid[i_f, i_a, imass, icol]
        # pts[1, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[1, 1] = ages[i_a]
        # pts[1, 2] = fehs[i_f]
        # vals[1] = grid[i_f, i_a, imass-1, icol]

        # i_f = ifeh - 1
        # i_a = iage 
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[2, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[2, 1] = ages[i_a]
        # pts[2, 2] = fehs[i_f]
        # vals[2] = grid[i_f, i_a, imass, icol]
        # pts[3, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[3, 1] = ages[i_a]
        # pts[3, 2] = fehs[i_f]
        # vals[3] = grid[i_f, i_a, imass-1, icol]

        # i_f = ifeh
        # i_a = iage - 1
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[4, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[4, 1] = ages[i_a]
        # pts[4, 2] = fehs[i_f]
        # vals[4] = grid[i_f, i_a, imass, icol]
        # pts[5, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[5, 1] = ages[i_a]
        # pts[5, 2] = fehs[i_f]
        # vals[5] = grid[i_f, i_a, imass-1, icol]

        # i_f = ifeh 
        # i_a = iage
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[6, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[6, 1] = ages[i_a]
        # pts[6, 2] = fehs[i_f]
        # vals[6] = grid[i_f, i_a, imass, icol]
        # pts[7, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[7, 1] = ages[i_a]
        # pts[7, 2] = fehs[i_f]
        # vals[7] = grid[i_f, i_a, imass-1, icol]
        
        # results[i] = interp_box(mass, age, feh, pts, vals)
        
    return results

@jit(nopython=True)
def interp_allcols(mass, age, feh, grid, mass_col, ages, fehs, grid_Ns, debug):
    Nage = len(ages)
    Nfeh = len(fehs)
    Ncols = grid.shape[3]

    results = np.zeros(Ncols, dtype=float64)

    ifeh = searchsorted(fehs, Nfeh, feh)
    iage = searchsorted(ages, Nage, age)
    if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
        for j in range(Ncols):
            results[j] = np.nan
        return results

    pts = np.zeros((8,3))
    vals = np.zeros((8,Ncols))

    i_f = ifeh - 1
    i_a = iage - 1
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[0, 0] = grid[i_f, i_a, imass, mass_col]
    pts[0, 1] = ages[i_a]
    pts[0, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[0, j] = grid[i_f, i_a, imass, j]
    pts[1, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[1, 1] = ages[i_a]
    pts[1, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[1, j] = grid[i_f, i_a, imass-1, j]

    i_f = ifeh - 1
    i_a = iage 
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[2, 0] = grid[i_f, i_a, imass, mass_col]
    pts[2, 1] = ages[i_a]
    pts[2, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[2, j] = grid[i_f, i_a, imass, j]
    pts[3, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[3, 1] = ages[i_a]
    pts[3, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[3] = grid[i_f, i_a, imass-1, j]

    i_f = ifeh
    i_a = iage - 1
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[4, 0] = grid[i_f, i_a, imass, mass_col]
    pts[4, 1] = ages[i_a]
    pts[4, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[4, j] = grid[i_f, i_a, imass, j]
    pts[5, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[5, 1] = ages[i_a]
    pts[5, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[5, j] = grid[i_f, i_a, imass-1, j]

    i_f = ifeh 
    i_a = iage
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[6, 0] = grid[i_f, i_a, imass, mass_col]
    pts[6, 1] = ages[i_a]
    pts[6, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[6, j] = grid[i_f, i_a, imass, j]
    pts[7, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[7, 1] = ages[i_a]
    pts[7, 2] = fehs[i_f]
    for j in range(Ncols):
        vals[7, j] = grid[i_f, i_a, imass-1, j]
    
    # if debug:
    #     result = np.zeros((8,4))
    #     for i in range(8):
    #         result[i, 0] = pts[i, 0]
    #         result[i, 1] = pts[i, 1]
    #         result[i, 2] = pts[i, 2]
    #         result[i, 3] = vals[i]
    #     return result
    # else:
    for j in range(Ncols):
        these_vals = np.zeros(8)
        for k in range(8):
            these_vals[k] = vals[k, j]

        results[j] = interp_box(mass, age, feh, pts, these_vals) 

    return results

@jit(nopython=True)
def interp_value(mass, age, feh, icol, 
                 grid, mass_col, ages, fehs, grid_Ns, debug):
                 # return_box):
    """mass, age, feh are *single values* at which values are desired

    icol is the column index of desired value
    grid is nfeh x nage x max(nmass) x ncols array
    mass_col is the column index of mass
    ages is grid of ages
    fehs is grid of fehs
    grid_Ns keeps track of nmass in each slice (beyond this are nans)
    
    TODO:  fix situation where there is exact match in age, feh, so we just
    interpolate along the track, not between...
    """
    

    Nage = len(ages)
    Nfeh = len(fehs)

    ifeh = searchsorted(fehs, Nfeh, feh)
    iage = searchsorted(ages, Nage, age)
    if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
        return np.nan

    pts = np.zeros((8,3))
    vals = np.zeros(8)

    i_f = ifeh - 1
    i_a = iage - 1
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[0, 0] = grid[i_f, i_a, imass, mass_col]
    pts[0, 1] = ages[i_a]
    pts[0, 2] = fehs[i_f]
    vals[0] = grid[i_f, i_a, imass, icol]
    pts[1, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[1, 1] = ages[i_a]
    pts[1, 2] = fehs[i_f]
    vals[1] = grid[i_f, i_a, imass-1, icol]

    i_f = ifeh - 1
    i_a = iage 
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[2, 0] = grid[i_f, i_a, imass, mass_col]
    pts[2, 1] = ages[i_a]
    pts[2, 2] = fehs[i_f]
    vals[2] = grid[i_f, i_a, imass, icol]
    pts[3, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[3, 1] = ages[i_a]
    pts[3, 2] = fehs[i_f]
    vals[3] = grid[i_f, i_a, imass-1, icol]

    i_f = ifeh
    i_a = iage - 1
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[4, 0] = grid[i_f, i_a, imass, mass_col]
    pts[4, 1] = ages[i_a]
    pts[4, 2] = fehs[i_f]
    vals[4] = grid[i_f, i_a, imass, icol]
    pts[5, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[5, 1] = ages[i_a]
    pts[5, 2] = fehs[i_f]
    vals[5] = grid[i_f, i_a, imass-1, icol]

    i_f = ifeh 
    i_a = iage
    Nmass = grid_Ns[i_f, i_a]
    imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    pts[6, 0] = grid[i_f, i_a, imass, mass_col]
    pts[6, 1] = ages[i_a]
    pts[6, 2] = fehs[i_f]
    vals[6] = grid[i_f, i_a, imass, icol]
    pts[7, 0] = grid[i_f, i_a, imass-1, mass_col]
    pts[7, 1] = ages[i_a]
    pts[7, 2] = fehs[i_f]
    vals[7] = grid[i_f, i_a, imass-1, icol]
    
    # if debug:
    #     result = np.zeros((8,4))
    #     for i in range(8):
    #         result[i, 0] = pts[i, 0]
    #         result[i, 1] = pts[i, 1]
    #         result[i, 2] = pts[i, 2]
    #         result[i, 3] = vals[i]
    #     return result
    # else:
    return interp_box(mass, age, feh, pts, vals)

@jit(nopython=True)
def interp_values_extinction(logg_arr, logT_arr, feh_arr, AV_arr, 
                 grid, loggs, logTs, fehs, AVs):
    """Just like interp_value_extinction but for arrays
    
    """
    
    N = len(logg_arr)
    results = np.zeros(N)

    for i in range(N):
        results[i] = interp_value_extinction(logg_arr[i], logT_arr[i], feh_arr[i], AV_arr[i], 
                                 grid, loggs, logTs, fehs, AVs)

    return results

@jit(nopython=True)
def interp_value_extinction(logg, logT, feh, AV, 
                     grid, loggs, logTs, fehs, AVs, normalize=True):

                 # return_box):
    """logg, logT, feh, AV are *single values* at which values are desired

    grid is nlogg x nlogT x nfeh x nAV array
    loggs is grid of loggs
    logTs is grid of logTs
    fehs is grid of fehs
    AVs is grid of AVs
    
    """

    # Check for nans.  If any of logg, logT, feh, AV are nan, return nan.
    if not ((logg < 0 or logg >=0) and
            (logT < 0 or logT >=0) and
            (feh < 0 or feh >=0) and
            (AV < 0 or AV >=0)):
        return np.nan

    Nlogg = len(loggs)
    NlogT = len(logTs)
    Nfeh = len(fehs)
    NAV = len(AVs)

    ilogg = searchsorted(loggs, Nlogg, logg)
    ilogT = searchsorted(logTs, NlogT, logT)
    ifeh = searchsorted(fehs, Nfeh, feh)
    iAV = searchsorted(AVs, NAV, AV)

    # If outside grid, extrapolate from closest
    if ilogg==0:
        ilogg = 1
    if ilogg==Nlogg:
        ilogg = Nlogg - 1

    if ilogT==0:
        ilogT = 1
    if ilogT==NlogT:
        ilogT = NlogT - 1

    if ifeh==0:
        ifeh = 1
    if ifeh==Nfeh:
        ifeh = Nfeh - 1

    if iAV==0:
        iAV = 1
    if iAV==NAV:
        iAV = NAV - 1

    pts = np.zeros((16,4))
    vals = np.zeros(16)

    irow = 0
    for i in range(2):
        i_g = ilogg - 1 + i
        for j in range(2):
            i_T = ilogT - 1 + j
            for k in range(2):
                i_f = ifeh - 1 + k
                for l in range(2):
                    i_A = iAV - 1 + l
                    pts[irow, 0] = loggs[i_g]
                    pts[irow, 1] = logTs[i_T]
                    pts[irow, 2] = fehs[i_f]
                    pts[irow, 3] = AVs[i_A]
                    vals[irow] = grid[i_g, i_T, i_f, i_A]
                    irow += 1

    if normalize:
        logg_norm = loggs[Nlogg - 1] - loggs[0]
        logT_norm = logTs[NlogT - 1] - logTs[0]
        feh_norm = fehs[Nfeh - 1] - fehs[0]
        AV_norm = 1. #AVs[NAV - 1] - AVs[0] # don't normalize AV

        return interp_box_4d(logg, logT, feh, AV, pts, vals,
                             logg_norm, logT_norm, feh_norm, AV_norm)
    else:
        return interp_box_4d(logg, logT, feh, AV, pts, vals)

    # return pts, vals

@jit(nopython=True)
def interp_box_4d(a, b, c, d, box, values, 
                  a_norm=1., b_norm=1., c_norm=1., d_norm = 1., 
                  p=-2):
    """
    box is nx4 array
    
    values is length-n array, corresponding to values at the "box" coords

    """
    
    # Calculate the distance to each vertex
    
    val = 0
    norm = 0
    N = box.shape[0]
    for i in range(N):
        # Inv distance, or Inv-dsq weighting
        if values[i] > 0.: # Extinction is positive; this should skip nans.
            dsq = (((a-box[i, 0])/a_norm)**2 + 
                   ((b-box[i, 1])/b_norm)**2 + 
                   ((c-box[i, 2])/c_norm)**2 + 
                   ((d-box[i, 3])/d_norm)**2)
            if dsq==0:
                return values[i]
            else:
                w = dsq**(p/2.)
                val += w * values[i]
                norm += w
    if norm==0:
        return np.nan
    else:
        return val/norm
 
