#!/usr/bin/env python


if __name__ == '__main__':
    import argparse

    import pandas as pd

    from isochrones.cluster import clusterfit


    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        comm = None
        rank = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('starfile', type=str, help='File with star cluster data.')
    parser.add_argument('--bands', default=None, nargs='*',
                        help='bands to use (vals, uncs must be in table)')
    parser.add_argument('--props', default=None, nargs='*',
                        help='properties to use (valus, uncs must be in table)')
    parser.add_argument('--models', type=str, default='mist')
    parser.add_argument('--max_distance', type=float, default=1000)
    parser.add_argument('--mineep', type=int, default=202)
    parser.add_argument('--maxeep', type=int, default=800)
    parser.add_argument('--maxAV', type=float, default=0.1)
    parser.add_argument('--overwrite', '-o', action='store_true')
    parser.add_argument('--nlive', type=int, default=1000)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--halo_fraction', type=float, default=0.5)
    parser.add_argument('--minq', type=float, default=0.2)

    args = parser.parse_args()

    clusterfit(args.starfile, bands=args.bands, props=args.props, models=args.models,
               max_distance=args.max_distance, mineep=args.mineep, maxeep=args.maxeep,
               maxAV=args.maxAV, overwrite=args.overwrite, nlive=args.nlive,
               name=args.name, halo_fraction=args.halo_fraction, minq=args.minq,
               comm=comm, rank=rank)
