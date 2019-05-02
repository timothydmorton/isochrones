import os, sys, re
import numpy as np
import pandas as pd
import logging
from multiprocessing import Pool

from .starmodel import StarModel


def get_quantiles(name, rootdir='.', columns=['eep','mass','radius','age','feh','distance','AV'],
                 qs=[0.05,0.16,0.5,0.84,0.95], modelname='mist_starmodel_single',
                 verbose=False, raise_exceptions=False):
    """Returns parameter quantiles for starmodel
    """

    modfile = os.path.join(rootdir, name,'{}.h5'.format(modelname))
    try:
        mod = StarModel.load_hdf(modfile)
    except:
        if verbose:
            print('cannnot load starmodel! ({})'.format(modfile))
        if raise_exceptions:
            raise
        return pd.DataFrame()

    # Get actual column names
    true_cols = []
    for c1 in mod.samples.columns:
        for c2 in columns:
            if re.search(c2, c1):
                true_cols.append(c1)

    q_df = mod.samples[true_cols].quantile(qs)

    df = pd.DataFrame(index=[name])
    for c in true_cols:
        for q in qs:
            col = c + '_{:02.0f}'.format(q*100)
            df.loc[name, col] = q_df.loc[q, c]

    return df

class quantile_worker(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, name):
        return get_quantiles(name, **self.kwargs)

def get_summary_df(names=None, pool=None, **kwargs):

    if pool is None:
        map_fn = map
    else:
        map_fn = pool.map

    worker = quantile_worker(**kwargs)
    dfs = map_fn(worker, names)

    df = pd.concat(dfs)
    return df

    if filename is None:
        filename = 'summary.h5'
    df.to_hdf(filename, 'df')
    pool.close()

    print('Summary dataframe written to {}'.format(filename))
    return df
