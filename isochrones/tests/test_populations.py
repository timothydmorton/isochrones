import unittest

from pandas.testing import assert_frame_equal
from scipy.stats import uniform, norm
from isochrones import get_ichrone
from isochrones.priors import ChabrierPrior, FehPrior, GaussianPrior, SalpeterPrior, DistancePrior, AVPrior
from isochrones.populations import StarFormationHistory, StarPopulation, BinaryDistribution, deredden


class PopulationTest(unittest.TestCase):
    def setUp(self):
        mist = get_ichrone("mist")
        mist.initialize()
        mist.track.initialize()

        sfh = StarFormationHistory()  # Constant SFR for 10 Gyr; or, e.g., dist=norm(3, 0.2)
        imf = SalpeterPrior(bounds=(0.4, 10))  # bounds on solar masses
        binaries = BinaryDistribution(fB=0.4, gamma=0.3)
        feh = GaussianPrior(-0.2, 0.2)
        distance = DistancePrior(max_distance=3000)  # pc
        AV = AVPrior(bounds=[0, 2])
        pop = StarPopulation(
            mist, sfh=sfh, imf=imf, feh=feh, distance=distance, binary_distribution=binaries, AV=AV
        )

        self.pop = pop
        self.mist = mist
        self.df = pop.generate(1000)
        self.dereddened_df = deredden(mist, self.df)

    def test_mags(self):
        """Check no total mags are null
        """
        mags = [f"{b}_mag" for b in self.mist.bands]
        assert self.df[mags].isnull().sum().sum() == 0

    def test_dereddening(self):
        """Check mass, age, feh the same when dereddened
        """

        cols = ["initial_mass", "initial_feh", "requested_age"]
        assert_frame_equal(self.df[cols], self.dereddened_df[cols])

        # Check de-reddening vis-a-vis A_x
        for b in self.mist.bands:
            diff = (self.dereddened_df[f"{b}_mag"] + self.df[f"A_{b}"]) - self.df[f"{b}_mag"]
            is_binary = self.df.mass_B > 0
            assert diff.loc[~is_binary].std() < 0.0001
