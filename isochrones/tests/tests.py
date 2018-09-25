
import tempfile
import os, shutil, glob
import logging

import numpy as np
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel, get_ichrone

def test_dartmouth_basic(bands='JHK'):
    dar = Dartmouth_Isochrone(bands)
    _basic_ic_checks(dar)

    assert np.isclose(dar.logg(73, 9.69897, -0.5), 4.4635)  # Grid point
    assert np.isclose(dar.logg(80, 9.3, 0.1), 3.9494549811008874)
    assert np.isclose(dar.logg(250, 9.32, 0.02), 1.5873518379443958)

def test_mist_basic(bands='JHK'):
    ic = MIST_Isochrone(bands)

    _basic_ic_checks(ic)

    assert np.isclose(ic.logg(683, 8.95, -1), 2.5158) # Grid point
    assert np.isclose(ic.logg(355, 9.653, 0.0), 4.412467501543451)
    assert np.isclose(ic.logg(700, 9.3, -0.03), 2.2354315457325225)

def test_spec(bands='JHK'):
    dar = get_ichrone('dartmouth', bands=bands)
    mist = get_ichrone('mist', bands=bands)
    _check_spec(dar)
    _check_spec(mist)

def test_AV():
    from isochrones.extinction import get_AV_infinity
    AV = get_AV_infinity(299.268036, 45.227428)
    assert np.isclose(AV, 1.216)

def test_get_ichrone(models=['dartmouth','mist'], bands='JHK'):
    for m in models:
        get_ichrone(m, bands=bands)

##########

def _basic_ic_checks(ic):
    age, feh = (9.5, -0.2)
    eep = ic.eep_from_mass(1.0, age, feh)
    assert np.isfinite(ic.radius(eep, age, feh))
    assert np.isfinite(ic.radius(np.ones(100)*eep, age, feh)).all()
    assert np.isfinite(ic.radius(eep, np.ones(100)*age, feh)).all()
    assert np.isfinite(ic.radius(eep, age, np.ones(100)*feh)).all()
    assert np.isfinite(ic.radius(eep, np.ones(100)*age, np.ones(100)*feh)).all()
    assert np.isfinite(ic.radius(np.ones(100)*eep, age, np.ones(100)*feh)).all()
    assert np.isfinite(ic.radius(np.ones(100)*eep, np.ones(100)*age, feh)).all()
    assert np.isfinite(ic.radius(np.ones(100)*eep, np.ones(100)*age, np.ones(100)*feh)).all()


    assert np.isfinite(ic.Teff(eep, age, feh))
    assert np.isfinite(ic.Teff(eep, np.ones(100)*age, feh)).all()

    assert np.isfinite(ic.density(eep, age, feh))
    assert np.isfinite(ic.density(eep, age, np.ones(100)*feh)).all()

    assert np.isfinite(ic.nu_max(eep, age, feh))
    assert np.isfinite(ic.delta_nu(eep, age, feh))

    args = (eep, age, feh, 500, 0.2)
    for b in ic.bands:
        assert np.isfinite(ic.mag[b](*args))

    # Make sure no ZeroDivisionError for on-the-grid calls (Github issue #64)
    ic.isochrone(8.0, feh=0.)

    # Make sure that passing nan returns nan (Github issue #65)
    assert np.isnan(ic.radius(1.0, np.nan, 0.1))


def _check_spec(ic, eep):
    mod = StarModel(ic, Teff=(5700,100), logg=(4.5, 0.1), feh=(0.0, 0.2))
    assert np.isfinite(mod.lnlike([eep, 9.6, 0.1, 200, 0.2]))

if __name__=='__main__':
    test_dartmouth_basic()
    test_mist_basic()
