
def test_age():
    from isochrones.priors import AgePrior
    age_prior = AgePrior()
    age_prior.test_integral()
    age_prior.test_sampling()


def test_distance():
    from isochrones.priors import DistancePrior
    distance_prior = DistancePrior()
    distance_prior.test_integral()
    distance_prior.test_sampling()


def test_AV():
    from isochrones.priors import AVPrior
    AV_prior = AVPrior()
    AV_prior.test_integral()
    AV_prior.test_sampling()


def test_q():
    from isochrones.priors import QPrior
    q_prior = QPrior()
    q_prior.test_integral()
    q_prior.test_sampling()


def test_salpeter():
    from isochrones.priors import SalpeterPrior
    salpeter_prior = SalpeterPrior()
    salpeter_prior.test_integral()
    salpeter_prior.test_sampling()


def test_feh():
    from isochrones.priors import FehPrior
    feh_prior = FehPrior()
    feh_prior.test_integral()
    feh_prior.test_sampling()
    feh_prior.bounds = (-3, 0.25)
    feh_prior.test_integral()
    feh_prior.test_sampling()
    assert feh_prior(-3.5) == 0
    assert feh_prior(0.4) == 0


def test_chabrier():
    from isochrones.priors import ChabrierPrior
    chabrier_prior = ChabrierPrior()
    chabrier_prior.test_integral()
    chabrier_prior.test_sampling()


