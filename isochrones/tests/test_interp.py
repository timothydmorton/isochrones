import itertools
import logging

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from isochrones.interp import DFInterpolator


def test_interp():
    xx, yy, zz = [np.arange(10 + np.log10(n))*n for n in [1, 10, 100]]

    def func(x, y, z):
        return x**2*np.cos(y/10) + z

    df = pd.DataFrame([(x, y, z, func(x, y, z)) for x, y, z in itertools.product(xx, yy, zz)],
                      columns=['x', 'y', 'z', 'val']).set_index(['x', 'y', 'z'])

    grid = np.reshape(df.val.values, (10, 11, 12))
    interp = RegularGridInterpolator([xx, yy, zz], grid)

    df_interp = DFInterpolator(df)

    grid_pars = [6., 50., 200.]
    pars = [3.1, 44., 503.]

    # Make sure grid point returns correct exact value
    assert df_interp(grid_pars, ['val']) == func(*grid_pars)

    # Check linear interpolation vis-a-vis scipy
    try:
        assert np.isclose(df_interp(pars, ['val']), interp(pars)[0], atol=1e-11)
    except AssertionError:
        logging.debug('mine: {}, scipy: {}'.format(df_interp(pars, ['val']), interp(pars)[0]))
        raise

    pts = np.random.random(size=(10, 3)) * 9
    pts[:, 1] *= 10
    pts[:, 2] *= 100

    assert np.allclose(df_interp([pts[:, 0], pts[:, 1], pts[:, 2]], ['val']).ravel(), interp(pts), atol=1e-11)
