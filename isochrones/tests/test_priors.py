from scipy.integrate import quad
import numpy as np


def test_age():
    from isochrones.priors import age_prior
    bounds = 9, 10.15
    _check_integral(age_prior, bounds)

def test_distance():
    from isochrones.priors import distance_prior
    bounds = 0, 3000
    _check_integral(distance_prior, bounds)

def test_AV():
    from isochrones.priors import AV_prior
    bounds = 0, 1.
    _check_integral(AV_prior, bounds)    

def test_q():
    from isochrones.priors import q_prior
    bounds = 0.1, 1.
    _check_integral(q_prior, bounds)    

def test_q():
    from isochrones.priors import salpeter_prior
    bounds = (0.1, 10)
    _check_integral(salpeter_prior, bounds)    

def test_feh():
    from isochrones.priors import feh_prior
    bounds = -np.inf, np.inf
    _check_integral(feh_prior, bounds) 

def _check_integral(pr_fn, bounds):
    def fn(x):
        return pr_fn(x, bounds=bounds)
    assert np.isclose(1, quad(fn, *bounds)[0])