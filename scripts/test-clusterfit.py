#!/usr/bin/env python
import argparse

from isochrones.cluster import StarClusterModel, simulate_cluster
from isochrones import get_ichrone
from isochrones.priors import FehPrior, FlatLogPrior


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0


parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=40)
parser.add_argument('--name', type=str, default='test-binary')
parser.add_argument('--age', type=float, default=8.56, help='log10(age [yr])')
parser.add_argument('--feh', type=float, default=-0.42)
parser.add_argument('--distance', type=float, default=200)
parser.add_argument('--AV', type=float, default=0.02, help='V-band extinction')
parser.add_argument('--alpha', type=float, default=-3, help='IMF index')
parser.add_argument('--gamma', type=float, default=0.3, help='mass ratio index')
parser.add_argument('--fB', type=float, default=0.5, help='binary fraction')
parser.add_argument('--models', type=str, default='mist')
parser.add_argument('--mineep', type=int, default=202)
parser.add_argument('--maxeep', type=int, default=605)
parser.add_argument('--maxAV', type=float, default=0.1)
parser.add_argument('--overwrite', '-o', action='store_true')
parser.add_argument('--nlive', type=int, default=1000)

args = parser.parse_args()


if rank == 0:
    ic = get_ichrone(args.models)

    pars = [args.age, args.feh, args.distance,
            args.AV, args.alpha, args.gamma, args.fB]
    print(pars)
    cat = simulate_cluster(args.N, *pars)
    print(cat.df.describe())

    cat.df.to_hdf('{}_stars.h5'.format(args.name), 'df')

    model = StarClusterModel(ic, cat, eep_bounds=(args.mineep, args.maxeep),
                             max_distance=args.distance*3, max_AV=args.maxAV, name=args.name)
    model.set_prior(feh=FehPrior(halo_fraction=0.5), age=FlatLogPrior((6, 9.5)))

    print('lnprior, lnlike, lnpost: {}'.format([model.lnprior(pars),
                                                model.lnlike(pars),
                                                model.lnpost(pars)]))

else:
    model = None

model = comm.bcast(model, root=0)
model.fit(overwrite=args.overwrite, n_live_points=args.nlive, init_MPI=False)
