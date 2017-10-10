from scipy.integrate import quad
import numpy as np
import logging


def test_age():
    from isochrones.priors import age_prior
    age_prior.test_integral()
    age_prior.test_sampling()

def test_distance():
    from isochrones.priors import distance_prior
    distance_prior.test_integral()
    distance_prior.test_sampling()

def test_AV():
    from isochrones.priors import AV_prior
    AV_prior.test_integral()
    AV_prior.test_sampling()

def test_q():
    from isochrones.priors import q_prior
    q_prior.test_integral()
    q_prior.test_sampling()

def test_salpeter():
    from isochrones.priors import salpeter_prior
    salpeter_prior.test_integral()
    salpeter_prior.test_sampling()

def test_feh():
    from isochrones.priors import feh_prior
    feh_prior.test_integral()
    feh_prior.test_sampling()
