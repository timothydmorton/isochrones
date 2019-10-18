"""
generate_cmd.py

Generates a table of N stars, with each row being a single star sampled from
the cluster population represented by a randomized choice of cluster parameter.

The cluster parameters are:

* age (drawn uniformly from 0 to 10 Gyr; stored in dataframe as log10(age))
* feh (metallicity)
* alpha (power-law index for mass function, and stars are drawn from Salpeter IMF)
* fB (binary fraction; sampled from 0 to 1)
* gamma (power-law index for binary mass ratio distribution)

The ranges in which these parameters are sampled are currently hard-coded.

The number of stars requested with -N will not be exactly the number of stars
generated; rather, the true number will be about 25% of that (because of stars
drawn outside their actual lifetimes).

The runtime of this script is about ~7 minutes for N=1e7.
"""
import pandas as pd
import numpy as np

from scipy.stats import uniform
from isochrones.priors import PowerLawPrior
from isochrones.utils import addmags
from isochrones import get_ichrone

mist = get_ichrone('mist')
age_dist = uniform(0, 10)  # age in Gyr
feh_dist = uniform(-2, 2.5)
alpha_dist = uniform(-3, 1)  # Salpeter slope
fB_dist = uniform(0, 1)
gamma_dist = uniform(0, 1)


def sample_params(N):
    return pd.DataFrame({'age': np.log10(1e9 * age_dist.rvs(N)),
                         'feh': feh_dist.rvs(N),
                         'alpha': alpha_dist.rvs(N),
                         'fB': fB_dist.rvs(N),
                         'gamma': gamma_dist.rvs(N)})


def sample_stars(ic, N, mass_range=(1, 10)):
    # Get params
    params = sample_params(N)

    # sample masses from PowerLaw
    masses = np.array([PowerLawPrior(a, bounds=mass_range).sample(1)[0] for a in params['alpha']])

    # Generate secondary stars
    is_binary = np.array([np.random.random() < fB for fB in params['fB']])
    qs = np.array([PowerLawPrior(gam, bounds=(0.1, 1)).sample(1)[0] for gam in params['gamma']])
    secondary_masses = masses * qs * is_binary

    primary_stars = ic.generate(masses, params['age'], params['feh'])
    secondary_stars = ic.generate(secondary_masses, params['age'], params['feh'])

    df_A = pd.DataFrame({'{}_A'.format(b): primary_stars['{}_mag'.format(b)] for b in ic.bands})
    df_B = pd.DataFrame({'{}_B'.format(b): secondary_stars['{}_mag'.format(b)] for b in ic.bands})

    for b in ic.bands:
        df_B.loc[np.logical_not(is_binary), '{}_B'.format(b)] = np.inf

    df_tot = pd.DataFrame({b: addmags(df_A['{}_A'.format(b)],
                                      df_B['{}_B'.format(b)]) for b in ic.bands})
    df_all = pd.concat([params, df_tot, df_A, df_B], axis=1)
    df_all['is_binary'] = is_binary
    df_all['mass'] = masses
    df_all['secondary_mass'] = secondary_masses
    return df_all.dropna(subset=[ic.bands[0]])


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', help='target number of stars', type=int)
    parser.add_argument('-o', '--output', help='output filename', default='cmd.hdf')

    args = parser.parse_args()

    mist = get_ichrone('mist')

    stars = sample_stars(mist, args.N)
    stars.to_hdf(args.output, 'df')

