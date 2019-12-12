import numpy as np
from scipy.stats import uniform

from .priors import ChabrierPrior, FehPrior, DistancePrior, PowerLawPrior
from .utils import addmags
from . import get_ichrone


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
        cdf = self.sfh_grid.cumsum()/self.sfh_grid.sum()
        u = np.random.random(N)
        i_bin = np.digitize(u, cdf)
        return np.log10(1e9 * self.t_grid[i_bin])


class BinaryDistribution(object):
    def __init__(self, fB=0.4, gamma=0.3, mass_ratio_distribution=None):
        self.fB = fB
        self.gamma = gamma
        if mass_ratio_distribution is None:
            mass_ratio_distribution = PowerLawPrior(self.gamma, bounds=(0.2, 1))
        self.mass_ratio_distribution = mass_ratio_distribution

    def sample(self, primary_masses):
        primary_masses = np.array(primary_masses)
        N = len(primary_masses)
        u = np.random.random(N)
        is_binary = u < self.fB
        q = self.mass_ratio_distribution.sample(N)
        secondary_mass = q * primary_masses * is_binary
        return secondary_mass


class StarPopulation(object):

    def __init__(self, ic,
                 sfh=StarFormationHistory(),
                 imf=ChabrierPrior(),
                 feh=FehPrior(),
                 binary_distribution=BinaryDistribution(),
                 distance=10.,
                 AV=0.):

        self._ic = ic
        self.sfh = sfh
        self.imf = imf
        self.feh = feh
        self.binary_distribution = binary_distribution
        self.distance = distance
        self.AV = AV

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    def generate(self, N, accurate=False, exact_N=True):
        N = int(N)
        masses = self.imf.sample(N)
        ages = self.sfh.sample_ages(N)
        fehs = self.feh.sample(N)

        if hasattr(self.distance, 'sample'):
            distances = self.distance.sample(N)
        else:
            distances = self.distance

        if hasattr(self.AV, 'sample'):
            AVs = self.AV.sample(N)
        else:
            AVs = self.AV

        population = self.ic.generate(masses, ages, fehs,
                                      distance=distances, AV=AVs)

        if exact_N:
            # Indices of null values
            bad_inds = population.isnull().sum(axis=1) > 0
            Nbad = bad_inds.sum()

            while Nbad > 0:
                masses = self.imf.sample(Nbad)
                ages = self.sfh.sample_ages(Nbad)
                fehs = self.feh.sample(Nbad)

                if hasattr(self.distance, 'sample'):
                    distances = self.distance.sample(N)
                else:
                    distances = self.distance

                if hasattr(self.AV, 'sample'):
                    AVs = self.AV.sample(N)
                else:
                    AVs = self.AV

                if Nbad == 1:
                    masses = masses[0]
                    ages = ages[0]
                    fehs = fehs[0]

                    try:
                        distances = distances[0]
                    except:
                        pass

                    try:
                        AVs = AVs[0]
                    except:
                        pass

                new_pop = self.ic.generate(masses, ages, fehs,
                                           distance=distances, AV=AVs)
                population.loc[bad_inds, :] = new_pop.values

                bad_inds = population.isnull().sum(axis=1) > 0
                Nbad = bad_inds.sum()

        population['distance'] = distances
        population['AV'] = AVs

        secondary_mass = self.binary_distribution.sample(population['mass'])
        secondary_population = self.ic.generate(secondary_mass,
                                                population['age'],
                                                population['feh'],
                                                distance=population['distance'],
                                                AV=population['AV'])

        population['mass_B'] = secondary_population['mass']
        for b in self.ic.bands:
            population[f'{b}_mag_A'] = population[f'{b}_mag'].copy()
            population[f'{b}_mag_B'] = secondary_population[f'{b}_mag']
            population.loc[population['mass_B'].isnull(), f'{b}_mag_B'] = np.inf
            population.loc[:, f'{b}_mag'] = addmags(population[f'{b}_mag_A'],
                                                    population[f'{b}_mag_B'])



        return population
