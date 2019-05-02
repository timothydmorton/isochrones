from .config import on_rtd

import os, re, sys
import warnings
import logging
import itertools
import pickle

try:
    import holoviews as hv
except ImportError:
    logging.warning('Holoviews not imported. Some visualizations will not be available.')
    pass

if not on_rtd:
    import pandas as pd
    import numpy as np

    from scipy.interpolate import LinearNDInterpolator as interpnd
    from scipy.optimize import newton, minimize
    import numpy.random as rand
    import matplotlib.pyplot as plt
    from numba import jit

    from astropy import constants as const

    # Define useful constants
    G = const.G.cgs.value
    MSUN = const.M_sun.cgs.value
    RSUN = const.R_sun.cgs.value

    # from .extinction import EXTINCTION, LAMBDA_EFF, extcurve, extcurve_0
    # from .interp import DFInterpolator, searchsorted, find_closest3
    # from .utils import polyval
    from .models import ModelGridInterpolator

else:
    G = 6.67e-11
    MSUN = 1.99e33
    RSUN = 6.96e10

from .config import ISOCHRONES
# from .grid import ModelGrid

def get_ichrone(models, bands=None, default=False, tracks=False, basic=False, **kwargs):
    """Gets Isochrone Object by name, or type, with the right bands

    If `default` is `True`, then will set bands
    to be the union of bands and default_bands
    """
    if not bands:
        bands = None

    if isinstance(models, ModelGridInterpolator):
        return models

    if type(models) is type(type):
        ichrone = models(bands)
    elif models=='mist':
        if tracks:
            from isochrones.mist import MIST_EvolutionTrack
            ichrone = MIST_EvolutionTrack(bands=bands, **kwargs)
        else:
            if basic:
                from isochrones.mist import MISTBasic_Isochrone
                ichrone = MISTBasic_Isochrone(bands=bands, **kwargs)
            else:
                from isochrones.mist import MIST_Isochrone
                ichrone = MIST_Isochrone(bands=bands, **kwargs)           
    else:
        raise ValueError('Unknown stellar models: {}'.format(models))
    return ichrone


