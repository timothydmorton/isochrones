import numpy as np
from scipy.stats import uniform

from .priors import ChabrierPrior, FehPrior, DistancePrior
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


class StarPopulation(object):

    def __init__(self, ic,
                 sfh=StarFormationHistory(),
                 imf=ChabrierPrior(),
                 feh=FehPrior(),
                 distance=10.,
                 AV=0.):

        self._ic = ic
        self.sfh = sfh
        self.imf = imf
        self.feh = feh
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

        return population
