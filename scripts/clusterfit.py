#!/usr/bin/env python
import argparse
import re

import pandas as pd

from isochrones.cluster import StarClusterModel, StarCatalog
from isochrones import get_ichrone
from isochrones.priors import FehPrior


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0


parser = argparse.ArgumentParser()
parser.add_argument('starfile', type=str, help='File with star cluster data.')
parser.add_argument('--bands', nargs='*', help='bands to use (vals, uncs must be in table)')
parser.add_argument('--props', nargs='*', help='properties to use (valus, uncs must be in table)')
parser.add_argument('--models', type=str, default='mist')
parser.add_argument('--max_distance', type=float, default=1000)
parser.add_argument('--mineep', type=int, default=202)
parser.add_argument('--maxeep', type=int, default=800)
parser.add_argument('--maxAV', type=float, default=0.1)
parser.add_argument('--overwrite', '-o', action='store_true')
parser.add_argument('--nlive', type=int, default=1000)

args = parser.parse_args()

if rank==0:
    ic = get_ichrone(args.models)

    if re.search('.h5', args.starfile) or re.search('.hdf', args.starfile):
        stars = pd.read_hdf(args.starfile)
    else:
        stars = pd.read_csv(args.starfile)

    cat = StarCatalog(stars)

    model = StarClusterModel(ic, cat, eep_bounds=(args.mineep, args.maxeep),
                             max_distance=args.max_distance, max_AV=args.maxAV, name=args.name)
    model.set_prior('feh', FehPrior(halo_fraction=0.5))

else:
    model = None

model = comm.bcast(model, root=0)
model.fit(overwrite=args.overwrite, n_live_points=args.nlive, init_MPI=False)
