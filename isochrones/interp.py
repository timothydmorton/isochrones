import os
import itertools
import logging

from numba import jit, float64, TypingError
from math import sqrt
import numpy as np
import pandas as pd

@jit(nopython=True)
def interp_box(x, y, z, box, values):
    """
    box is 8x3 array, though not really a box

    values is length-8 array, corresponding to values at the "box" coords

    TODO: should make power `p` an argument
    """

    # Calculate the distance to each vertex

    val = 0
    norm = 0
    for i in range(8):
        # Inv distance, or Inv-dsq weighting
        distance = sqrt((x-box[i,0])**2 + (y-box[i,1])**2 + (z-box[i, 2])**2)

        # If you happen to land on exactly a corner, you're done.
        if distance == 0:
            val = values[i]
            norm = 1.
            break

        w = 1./distance
        # w = 1./((x-box[i,0])*(x-box[i,0]) +
        #         (y-box[i,1])*(y-box[i,1]) +
        #         (z-box[i, 2])*(z-box[i, 2]))
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
    if np.isnan(x1) or np.isnan(x2) or np.isnan(x3):
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

    def __call__(self, p, col):
        icol = self.columns.index(col)
        args = (*p, self.grid, icol, *self.index_columns)

        if isinstance(p[0], float) and isinstance(p[1], float) and isinstance(p[2], float):
            return interp_value(*args)

        else:
            b = np.broadcast(*p)
            pp = [np.resize(x, b.shape).astype(float) for x in p]

            return interp_values(*pp, self.grid, icol, *self.index_columns)
