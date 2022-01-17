import unittest

from pandas.testing import assert_frame_equal
from isochrones import get_ichrone
from isochrones.priors import GaussianPrior, SalpeterPrior, DistancePrior, AVPrior
from isochrones.populations import StarFormationHistory, StarPopulation, deredden


def old_deredden(ic, pop, accurate=False, **kwargs):
    """Old version of deredden that regenerates population from scratch with AV=0
    """

    return ic.generate_binary(
        pop["initial_mass_0"].values,
        pop["initial_mass_1"].values,
        pop["requested_age_0"].values,
        pop["initial_feh_0"].values,
        distance=pop["distance_0"].values,
        AV=0.0,
        all_As=True,
        accurate=accurate,
        **kwargs,
    )


class PopulationTest(unittest.TestCase):
    def setUp(self):
        mist = get_ichrone("mist")

        sfh = StarFormationHistory()  # Constant SFR for 10 Gyr; or, e.g., dist=norm(3, 0.2)
        imf = SalpeterPrior(bounds=(0.4, 10))  # bounds on solar masses
        fB = 0.4
        gamma = 0.3
        feh = GaussianPrior(-0.2, 0.2)
        distance = DistancePrior(max_distance=3000)  # pc
        AV = AVPrior(bounds=[0, 2])
        pop = StarPopulation(mist, imf=imf, fB=fB, gamma=gamma, sfh=sfh, feh=feh, distance=distance, AV=AV)

        self.pop = pop
        self.mist = mist
        self.df = pop.generate(1000)
        self.dereddened_df = deredden(self.df)

    def test_old_deredden(self):
        """Test dereddening against version that regenerates population
        """
        old_dereddened_df = old_deredden(self.mist, self.df)

        assert_frame_equal(self.dereddened_df.fillna(0), old_dereddened_df.fillna(0))

    def test_mags(self):
        """Check no total mags are null
        """
        mags = [f"{b}_mag" for b in self.mist.bands]
        assert self.df[mags].isnull().sum().sum() == 0

    def test_dereddening(self):
        """Check mass, age, feh the same when dereddened
        """

        cols = ["initial_mass_0", "initial_feh_0", "requested_age_0"]
        assert_frame_equal(self.df[cols], self.dereddened_df[cols])

        # Check de-reddening vis-a-vis A_x
        for b in self.mist.bands:
            diff = (self.dereddened_df[f"{b}_mag"] + self.df[f"A_{b}_0"]) - self.df[f"{b}_mag"]
            is_binary = self.df.mass_1 > 0
            assert diff.loc[~is_binary].std() < 0.0001

    def test_extinction(self):
        from numpy.testing import assert_array_almost_equal

        from isochrones.utils import addmags
        import numpy as np

        assert_array_almost_equal(
            self.df["G_mag"],
            addmags(
                self.dereddened_df["G_mag_0"] + self.df["A_G_0"],
                (self.dereddened_df["G_mag_1"] + self.df["A_G_1"]).fillna(np.inf),
            ),
        )

    def test_generate(self):
        """Make sure corner case when regenerating 1 doesn't break.
        """
        for i in range(10):
            self.pop.generate(10)
