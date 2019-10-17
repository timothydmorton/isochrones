from isochrones.starmodel import StarModel, BasicStarModel
from isochrones import get_ichrone
import numpy as np

mist = get_ichrone('mist')

props = dict(Teff=(5800, 100), logg=(4.5, 0.1),
             J=(3.58, 0.05), K=(3.22, 0.05),
             parallax=(100, 0.1))


props_phot = dict(J=(3.58, 0.05), K=(3.22, 0.05), parallax=(100, 0.1))
props_spec = dict(Teff=(5800, 100), logg=(4.5, 0.1), parallax=(100, 0.1))


def test_compare_starmodels(props=props):
    m1 = StarModel(mist, **props)
    m2 = BasicStarModel(mist, **props)

    # Ensure priors are identical
    for k in ['mass', 'feh', 'age', 'distance', 'AV', 'eep']:
        m2.set_prior(**{k: m1._priors[k]})

    pars = [300, 9.8, 0.01, 100, 0.1]
    assert np.isclose(m1.lnlike(pars), m2.lnlike(pars))
    assert np.isclose(m1.lnprior(pars), m2.lnprior(pars))
    assert np.isclose(m1.lnpost(pars), m2.lnpost(pars))

    m1_bin = StarModel(mist, **props, N=2)
    m2_bin = BasicStarModel(mist, **props, N=2)

    # Ensure priors are identical
    for k in ['mass', 'feh', 'age', 'distance', 'AV', 'eep']:
        m2_bin.set_prior(**{k: m1_bin._priors[k]})

    pars = [300, 280, 9.8, 0.01, 100, 0.1]
    assert np.isclose(m1_bin.lnlike(pars), m2_bin.lnlike(pars))
    assert np.isclose(m1_bin.lnprior(pars), m2_bin.lnprior(pars))
    assert np.isclose(m1_bin.lnpost(pars), m2_bin.lnpost(pars))

    m1_trip = StarModel(mist, **props, N=3)
    m2_trip = BasicStarModel(mist, **props, N=3)

    # Ensure priors are identical
    for k in ['mass', 'feh', 'age', 'distance', 'AV', 'eep']:
        m2_trip.set_prior(**{k: m1_trip._priors[k]})

    pars = [300, 280, 260., 9.8, 0.01, 100, 0.1]
    assert np.isclose(m1_trip.lnlike(pars), m2_trip.lnlike(pars))
    assert np.isclose(m1_trip.lnprior(pars), m2_trip.lnprior(pars))
    assert np.isclose(m1_trip.lnpost(pars), m2_trip.lnpost(pars))


def test_compare_spec():
    test_compare_starmodels(props_spec)


def test_compare_phot():
    test_compare_starmodels(props_phot)
