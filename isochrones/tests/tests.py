
import tempfile
import os, shutil, glob
import logging

import numpy as np
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel, get_ichrone

DAR = Dartmouth_Isochrone()
MIST = MIST_Isochrone()

def test_dartmouth_basic(bands=['J']):
    dar = Dartmouth_Isochrone(bands)
    _basic_ic_checks(dar)

    assert np.isclose(dar.radius(1., 9.5, 0.0), 0.95409593462955189)
    assert np.isclose(dar.radius(1.01, 9.72, 0.02), 1.0435559519926865)
    assert np.isclose(dar.radius(1.21, 9.38, 0.11), 1.2762022494652547)
    assert np.isclose(dar.radius(0.61, 9.89, -0.22), 0.5964945760402941)

def test_mist_basic(bands=['J']):
    ic = MIST_Isochrone(bands, version='1.0')
    ic2 = MIST_Isochrone(bands + ['TESS', 'BP', 'RP'], version='1.1')

    _basic_ic_checks(ic)
    _basic_ic_checks(ic2)

    assert np.isclose(ic.radius(1.0, 9.5, 0.0), 0.9764494078461442)
    assert np.isclose(ic.radius(1.01, 9.72, 0.02), 1.0671791635014685)
    assert np.isclose(ic.radius(1.21, 9.38, 0.11), 1.2963342261673843)
    assert np.isclose(ic.radius(0.61, 9.89, -0.22), 0.5873830516268735)

    assert np.isclose(ic2.radius(1.0, 9.5, 0.0), 0.9765234978729515)
    assert np.isclose(ic2.radius(1.01, 9.72, 0.02), 1.0671845393364638)
    assert np.isclose(ic2.radius(1.21, 9.38, 0.11), 1.2963536270911573)
    assert np.isclose(ic2.radius(0.61, 9.89, -0.22), 0.5873849015685695)


def test_spec():
    _check_spec(DAR)
    _check_spec(MIST)

def test_AV():
    from isochrones.extinction import get_AV_infinity
    AV = get_AV_infinity(299.268036, 45.227428)
    assert np.isclose(AV, 1.216)

def test_get_ichrone(models=['dartmouth','dartmouthfast','mist'], bands=['J']):
    for m in models:
        get_ichrone(m, bands=bands)

##########

def _basic_ic_checks(ic):
    mass, age, feh = (1., 9.5, -0.2)
    assert np.isfinite(ic.radius(mass, age, feh))
    assert np.isfinite(ic.radius(np.ones(100)*mass, age, feh)).all()
    assert np.isfinite(ic.radius(mass, np.ones(100)*age, feh)).all()
    assert np.isfinite(ic.radius(mass, age, np.ones(100)*feh)).all()
    assert np.isfinite(ic.radius(mass, np.ones(100)*age, np.ones(100)*feh)).all()
    assert np.isfinite(ic.radius(np.ones(100)*mass, age, np.ones(100)*feh)).all()
    assert np.isfinite(ic.radius(np.ones(100)*mass, np.ones(100)*age, feh)).all()
    assert np.isfinite(ic.radius(np.ones(100)*mass, np.ones(100)*age, np.ones(100)*feh)).all()


    assert np.isfinite(ic.Teff(mass, age, feh))
    assert np.isfinite(ic.Teff(mass, np.ones(100)*age, feh)).all()

    assert np.isfinite(ic.density(mass, age, feh))
    assert np.isfinite(ic.density(mass, age, np.ones(100)*feh)).all()

    assert np.isfinite(ic.nu_max(mass, age, feh))
    assert np.isfinite(ic.delta_nu(mass, age, feh))

    args = (mass, age, feh, 500, 0.2)
    for b in ic.bands:
        assert np.isfinite(ic.mag[b](*args))

    # Make sure no ZeroDivisionError for on-the-grid calls (Github issue #64)
    ic.isochrone(8.0, feh=0.)

    # Make sure that passing nan returns nan (Github issue #65)
    assert np.isnan(ic.radius(1.0, np.nan, 0.1))


def _check_spec(ic):
    mod = StarModel(ic, Teff=(5700,100), logg=(4.5, 0.1), feh=(0.0, 0.2))
    assert np.isfinite(mod.lnlike([1.0, 9.6, 0.1, 200, 0.2]))

if __name__=='__main__':
    test_dartmouth_basic()
    test_mist_basic()
