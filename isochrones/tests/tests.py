
import tempfile
import os, shutil, glob
import logging

import numpy as np
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel

DAR = Dartmouth_Isochrone()
MIST = MIST_Isochrone()

def test_dartmouth_basic(bands=['z', 'B', 'W3', 'LSST_r', 'J', 'UK_J']):
    dar = Dartmouth_Isochrone(bands)
    _basic_ic_checks(dar)

def test_mist_basic(bands=['G','B','V','J','H','K','W1','W2','W3']):
    ic = MIST_Isochrone(bands)
    _basic_ic_checks(ic)

def test_spec():
    _check_spec(DAR)
    _check_spec(MIST)

def test_AV():
    from isochrones.extinction import get_AV_infinity
    AV = get_AV_infinity(299.268036, 45.227428)
    assert np.isclose(AV, 1.216)

##########  

def _basic_ic_checks(ic):
    ic.radius(1., 9.5, -0.2)
    ic.radius(np.ones(100), 9.5, 0.0)
    args = (1, 9.5, 0.1, 500, 0.2)
    for b in ic.bands:
        ic.mag[b](*args)

def _check_spec(ic):
    mod = StarModel(ic, Teff=(5700,100), logg=(4.5, 0.1), feh=(0.0, 0.2))
    assert np.isfinite(mod.lnlike([1.0, 9.6, 0.1, 200, 0.2]))

