import os
import itertools
import logging

from numba import jit, float64, TypingError, typeof
from math import sqrt
import numpy as np
import pandas as pd

@jit(nopython=True)
def interp_box(x, y, z, box, values):
    """
    box is 8x3 array, though not really a box

    Implementing trilinear interpolation according to

    https://en.wikipedia.org/wiki/Trilinear_interpolation

    Corners are organized as follows in following order in 'box' and 'values'

    000
    001
    010
    011
    100
    101
    110
    111

    values is length-8 array, corresponding to values at the "box" coords

    TODO: should make power `p` an argument
    """

    # Calculate the distance to each vertex

    x0 = box[0, 0]
    x1 = box[4, 0]
    y0 = box[0, 1]
    y1 = box[2, 1]
    z0 = box[0, 2]
    z1 = box[1, 2]

    if x1 == x0:
        xd = 0
    else:
        xd = (x - x0) / (x1 - x0)

    if y1 == y0:
        yd = 0
    else:
        yd = (y - y0) / (y1 - y0)

    if z1 == z0:
        zd = 0
    else:
        zd = (z - z0) / (z1 - z0)

    c000 = values[0]
    c001 = values[1]
    c010 = values[2]
    c011 = values[3]
    c100 = values[4]
    c101 = values[5]
    c110 = values[6]
    c111 = values[7]

    # Replace nans with zeros, so they disappear in calculations
    if c000 != c000:
        c000 = 0.
    if c001 != c001:
        c001 = 0.
    if c010 != c010:
        c010 = 0.
    if c011 != c011:
        c011 = 0.
    if c100 != c100:
        c100 = 0.
    if c101 != c101:
        c101 = 0.
    if c110 != c110:
        c110 = 0.
    if c111 != c111:
        c111 = 0.

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    if c == 0.:
        return np.nan
    else:
        return c

@jit(nopython=True)
def searchsorted(arr, N, x):
    """N is length of arr
    """
    L = 0
    R = N-1
    done = False
    eq = False
    m = (L+R)//2
    while not done:
        if arr[m] < x:
            L = m + 1
        elif arr[m] > x:
            R = m - 1
        elif arr[m] == x:
            L = m
            eq = True
            done = True
        m = (L+R)//2
        if L>R:
            done = True
    return L, eq

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
def interp_values(xx1, xx2, xx3,
                 grid, icol,
                 ii1, ii2, ii3):
    """xx1, xx2, xx3 are all arrays at which values are desired


    """

    N = len(xx1)
    results = np.zeros(N)

    for i in range(N):
        results[i] = interp_value(xx1[i], xx2[i], xx3[i],
                                  grid, icol,
                                  ii1, ii2, ii3)

    return results

@jit(nopython=True)
def interp_value(x1, x2, x3,
                 grid, icol,
                 ii1, ii2, ii3):
    """x1, x2, x3 are *single values* at which values in val_col are desired

    """
    if ((not x1 < 0 and not x1 >= 0) or
        (not x2 < 0 and not x2 >= 0) or
        (not x3 < 0 and not x3 >= 0)):
        return np.nan

    n1 = len(ii1)
    n2 = len(ii2)
    n3 = len(ii3)

    i1, eq1 = searchsorted(ii1, n1, x1)
    i2, eq2 = searchsorted(ii2, n2, x2)
    i3, eq3 = searchsorted(ii3, n3, x3)

    if (i1==0 or i2==0 or i3==0 or
        i1==n1 or i2==n2 or i3==n3):
        return np.nan

    pts = np.zeros((8,3))
    vals = np.zeros(8)

    i_1 = i1 - 1
    i_2 = i2 - 1
    i_3 = i3 - 1
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[0, 0] = ii1[i_1]
    pts[0, 1] = ii2[i_2]
    pts[0, 2] = ii3[i_3]
    vals[0] = grid[i_1, i_2, i_3, icol]

    i_1 = i1 - 1
    i_2 = i2 - 1
    i_3 = i3
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[1, 0] = ii1[i_1]
    pts[1, 1] = ii2[i_2]
    pts[1, 2] = ii3[i_3]
    vals[1] = grid[i_1, i_2, i_3, icol]

    i_1 = i1 - 1
    i_2 = i2
    i_3 = i3 - 1
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[2, 0] = ii1[i_1]
    pts[2, 1] = ii2[i_2]
    pts[2, 2] = ii3[i_3]
    vals[2] = grid[i_1, i_2, i_3, icol]

    i_1 = i1 - 1
    i_2 = i2
    i_3 = i3
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[3, 0] = ii1[i_1]
    pts[3, 1] = ii2[i_2]
    pts[3, 2] = ii3[i_3]
    vals[3] = grid[i_1, i_2, i_3, icol]

    i_1 = i1
    i_2 = i2 - 1
    i_3 = i3 - 1
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[4, 0] = ii1[i_1]
    pts[4, 1] = ii2[i_2]
    pts[4, 2] = ii3[i_3]
    vals[4] = grid[i_1, i_2, i_3, icol]

    i_1 = i1
    i_2 = i2 - 1
    i_3 = i3
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[5, 0] = ii1[i_1]
    pts[5, 1] = ii2[i_2]
    pts[5, 2] = ii3[i_3]
    vals[5] = grid[i_1, i_2, i_3, icol]

    i_1 = i1
    i_2 = i2
    i_3 = i3 - 1
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[6, 0] = ii1[i_1]
    pts[6, 1] = ii2[i_2]
    pts[6, 2] = ii3[i_3]
    vals[6] = grid[i_1, i_2, i_3, icol]

    i_1 = i1
    i_2 = i2
    i_3 = i3
    if eq1:
        i_1 = i1
    if eq2:
        i_2 = i2
    if eq3:
        i_3 = i3
    pts[7, 0] = ii1[i_1]
    pts[7, 1] = ii2[i_2]
    pts[7, 2] = ii3[i_3]
    vals[7] = grid[i_1, i_2, i_3, icol]


#     result = np.zeros((8,4))
#     for i in range(8):
#         result[i, 0] = pts[i, 0]
#         result[i, 1] = pts[i, 1]
#         result[i, 2] = pts[i, 2]
#         result[i, 3] = vals[i]
#     return result
    return interp_box(x1, x2, x3, pts, vals)

@jit(nopython=True)
def sign(x):
    if x < 0:
        return -1
    else:
        return 1

# @jit(nopython=True)
def find_closest3(val, a, b,
                  v1, v2,
                  grid, icol,
                  ii1, ii2, ii3,
                  bisect_tol=0.5, newton_tol=0.01,
                  max_iter=100, debug=False):
    """Find value of 3rd index array where interp_value is closest to val

    val : value to match
    a, b: min and max x-value to serch
=   x1, x2 : first and second values to pass to inter_values
    grid : 4d grid
    icol : index of value dimension of grid
    ii1, ii2, ii3 : grid dimension arrays
    """

    # First, do a bisect search to get it close
    done = False
    ya = interp_value(v1, v2, a, grid, icol, ii1, ii2, ii3) - val
    yb = interp_value(v1, v2, b, grid, icol, ii1, ii2, ii3) - val
    if debug:
        print('Initial values: {}: {}'.format((a, b), (ya, yb)))

    if yb != yb or ya != ya:
        # bounds are nan, return nan.
        return np.nan
    elif abs(ya) < newton_tol:
        return float(a)
    elif abs(yb) < newton_tol:
        return float(b)
    elif ya > 0 and yb > 0:
        return np.nan
    elif yb < 0 and yb < 0:
        return np.nan

    else:
        if debug:
            print('doing bisect search...')

        while not done:
            c = (a + b) / 2
            yc = interp_value(v1, v2, c, grid, icol, ii1, ii2, ii3) - val
            if yc == 0 or (b - a) / 2 < bisect_tol:
                done = True
            if sign(yc) == sign(ya): # (yc >= 0 and ya >= 0) or (yc < 0 and ya < 0):
                a = c
                ya = yc
            else:
                b = c
                yb = yc
            if debug:
                print('{0} {1}'.format((a,b,c), (ya,yb,yc)))

    # Now, use the value at index c to seed Newton-secant algorithm
    tol = 1000.
    i = 0
    x0 = c
    y0 = yc
    x1 = x0 + 0.1
    y1 = interp_value(v1, v2, x1, grid, icol, ii1, ii2, ii3) - val

    if debug:
        print('Newton-secant method...')
    while tol > newton_tol and i < max_iter:
        newx = (x0 * y1 - x1 * y0) / (y1 - y0)
        x0 = x1
        y0 = y1
        x1 = newx
        y1 = interp_value(v1, v2, x1, grid, icol, ii1, ii2, ii3) - val

        # Boo!
        while not y1 == y1:
            if debug:
                print('{0} {1}'.format(x1, y1))
            raise RuntimeError('ran into nan.' +
                               'run {} with debug=True to see why.'.format((val, v2, v1)))

        if y1 >= 0:
            tol = y1
        else:
            tol = -y1
        i += 1
        if debug:
            print('{0} {1}'.format(x1, y1))

    return x1


class DFInterpolator(object):
    """Interpolate column values of DataFrame with full-grid hierarchical index

    """
    def __init__(self, df, filename=None, recalc=False):

        self.filename = filename
        self.grid = self._make_grid(df, recalc=recalc)
        self.columns = list(df.columns)
        self.index_columns = tuple(np.array(l, dtype=float) for l in df.index.levels)
        self.index_names = df.index.names

    def _make_grid(self, df, recalc=False):
        if self.filename is not None and os.path.exists(self.filename) and not recalc:
            grid = np.load(self.filename)
        else:
            idx = pd.MultiIndex.from_tuples([ixs for ixs in itertools.product(*df.index.levels)])

            # Make an empty dataframe with the completely gridded index, and fill
            grid_df = pd.DataFrame(index=idx, columns=df.columns)
            grid_df.loc[df.index] = df
            shape = [len(l) for l in df.index.levels] + [len(df.columns)]

            grid = np.array(grid_df.values, dtype=float).reshape(shape)

            if self.filename is not None:
                np.save(self.filename, grid)

        return grid

    def find_closest(self, val, lo, hi, v1, v2,
                     col='initial_mass', debug=False):
        icol = self.columns.index(col)

        return find_closest3(val, lo, hi, v1, v2,
                             self.grid, icol,
                             *self.index_columns, debug=debug)

    def __call__(self, p, col):
        icol = self.columns.index(col)
        args = (*p, self.grid, icol, *self.index_columns)

        if ((isinstance(p[0], float) or isinstance(p[0], int)) and
            (isinstance(p[1], float) or isinstance(p[1], int)) and
            (isinstance(p[2], float) or isinstance(p[2], int))):
            return interp_value(*args)

        else:
            b = np.broadcast(*p)
            pp = [np.resize(x, b.shape).astype(float) for x in p]

            return interp_values(*pp, self.grid, icol, *self.index_columns)
