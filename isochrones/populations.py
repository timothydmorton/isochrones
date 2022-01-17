import re

import numpy as np
from scipy.stats import uniform

from .priors import ChabrierPrior, FehPrior, PowerLawPrior


class StarFormationHistory(object):
    """Star formation history

    Parameters
    ----------
    dist : `scipy.stats.distribution`
        Probability distribution for stellar ages; equivalent to a
        normalized dM/dT SF history.
    """

    def __init__(self, dist=None):
        if dist is None:
            dist = uniform(0, 10)  # default: uniform distribution 0 to 10 Gyr
        self.dist = dist

    def sample_ages(self, N):
        return np.log10(1e9 * self.dist.rvs(N))


class StarFormationHistoryGrid(StarFormationHistory):
    """ SFH defined in arbitrary time bins
    """

    def __init__(self, t_grid, sfh_grid):
        self.t_grid = t_grid
        self.sfh_grid = sfh_grid

    def sample_ages(self, N):
        """Sample N stellar ages from SFH
        """
        cdf = self.sfh_grid.cumsum() / self.sfh_grid.sum()
        u = np.random.random(N)
        i_bin = np.digitize(u, cdf)
        return np.log10(1e9 * self.t_grid[i_bin])


class BinaryDistribution(object):
    def __init__(self, imf, fB=0.4, gamma=0.3, mass_ratio_distribution=None):
        self.imf = imf
        self.fB = fB
        self.gamma = gamma
        if mass_ratio_distribution is None:
            mass_ratio_distribution = PowerLawPrior(self.gamma, bounds=(0.2, 1))
        self.mass_ratio_distribution = mass_ratio_distribution

    def sample(self, N):
        primary_mass = self.imf.sample(N)
        u = np.random.random(N)
        is_binary = u < self.fB
        q = self.mass_ratio_distribution.sample(N)
        secondary_mass = q * primary_mass * is_binary
        return primary_mass, secondary_mass


class StarPopulation(object):
    def __init__(
        self,
        ic,
        imf=ChabrierPrior(),
        fB=0.4,
        gamma=0.3,
        sfh=StarFormationHistory(),
        feh=FehPrior(),
        mass_ratio_distribution=None,
        distance=10.0,
        AV=0.0,
    ):

        self._ic = ic
        self.sfh = sfh
        self.imf = imf
        self.fB = fB
        self.gamma = gamma
        self.binary_distribution = BinaryDistribution(
            imf, fB=fB, gamma=gamma, mass_ratio_distribution=mass_ratio_distribution
        )
        self.feh = feh
        self.distance = distance
        self.AV = AV

    @property
    def ic(self):
        if type(self._ic) == type:
            self._ic = self._ic()
        return self._ic

    def generate(self, N, accurate=False, exact_N=True, **kwargs):
        N = int(N)
        masses, secondary_masses = self.binary_distribution.sample(N)
        ages = self.sfh.sample_ages(N)
        fehs = self.feh.sample(N)

        if hasattr(self.distance, "sample"):
            distances = self.distance.sample(N)
        else:
            distances = self.distance

        if hasattr(self.AV, "sample"):
            AVs = self.AV.sample(N)
        else:
            AVs = self.AV

        population = self.ic.generate_binary(
            masses,
            secondary_masses,
            ages,
            fehs,
            distance=distances,
            AV=AVs,
            all_As=True,
            accurate=accurate,
            **kwargs,
        )

        if exact_N:
            # Indices of null values
            bad_inds = population.mass_0.isnull()
            Nbad = bad_inds.sum()

            while Nbad > 0:
                new_masses, new_secondary_masses = self.binary_distribution.sample(Nbad)
                new_ages = self.sfh.sample_ages(Nbad)
                new_fehs = self.feh.sample(Nbad)

                if hasattr(self.distance, "sample"):
                    new_distances = self.distance.sample(Nbad)
                else:
                    new_distances = self.distance

                if hasattr(self.AV, "sample"):
                    new_AVs = self.AV.sample(Nbad)
                else:
                    new_AVs = self.AV

                new_pop = self.ic.generate_binary(
                    new_masses,
                    new_secondary_masses,
                    new_ages,
                    new_fehs,
                    distance=new_distances,
                    AV=new_AVs,
                    all_As=True,
                    accurate=accurate,
                    **kwargs,
                )

                population.loc[bad_inds, :] = new_pop.values
                ages[bad_inds] = new_ages

                bad_inds = population.mass_0.isnull()
                Nbad = bad_inds.sum()

        else:
            population = population.dropna(subset=["mass_0"])

        return population


def deredden(pop, accurate=False, **kwargs):
    """Returns the dereddened version of the population (AV=0)

    Parameters
    ----------
    pop : pandas.DataFrame
        DataFrame of stars, including (at least) mass


    Returns
    -------
    new_pop : pandas.DataFrame
        All the same stars as input, but with AV=0
    """
    new_pop = pop.copy()

    # All columns that end in _mag, with those last four chars removed
    bands = [c[:-4] for c in pop.columns if re.search(r"(\w+)_mag$", c)]

    new_pop["AV_0"] = 0.0
    new_pop["AV_1"] = 0.0

    for b in bands:
        new_pop[f"{b}_mag"] -= new_pop[f"A_{b}"]
        new_pop[f"{b}_mag_0"] -= new_pop[f"A_{b}_0"]
        new_pop[f"{b}_mag_1"] -= new_pop[f"A_{b}_1"]

        new_pop[f"A_{b}"] = 0.0
        new_pop[f"A_{b}_0"] = 0.0
        new_pop[f"A_{b}_1"] = 0.0

    return new_pop
