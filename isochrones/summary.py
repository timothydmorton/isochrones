import re, os
import numpy as np
import pandas as pd
import pdb
import logging
from multiprocessing import Pool

def get_quantiles(name, directory, columns=['mass','radius','Teff',
                                'logg', 'age','feh','distance','AV'],
                 qs=[0.05,0.16,0.5,0.84,0.95],
                 verbose=False, raise_exceptions=False):
    """Returns parameter quantiles for StarModel saved as '<directory>/<name>.h5'
    """
    
    h5filename = os.path.join(directory, '{}.h5'.format(name))
    try:
        samples = pd.read_hdf(h5filename, 'samples')
    except KeyError:
        logging.error('Cannot load "samples" from {}'.format(h5filename))
        return pd.DataFrame()

    # Get actual column names    
    true_cols = []
    for c1 in samples.columns:
        for c2 in columns:
            if re.search(c2, c1):
                true_cols.append(c1)

    q_df = samples[true_cols].quantile(qs)

    try:
        name = int(name)
    except:
        pass

    df = pd.DataFrame(index=[name])

    for c in true_cols:
        for q in qs:
            col = c + '_{:02.0f}'.format(q*100)
            df.ix[name, col] = q_df.ix[q, c]

    return df

class quantile_worker(object):
    def __init__(self, directory, **kwargs):
        self.directory = directory
        self.kwargs = kwargs

    def __call__(self, i):
        return get_quantiles(i, self.directory, **self.kwargs)

def make_summary_df(directory, processes=1, filename=None, **kwargs):
    """Makes summary quantile df for all starmodels (*.h5) in directory 
    """
    filenames = [f for f in os.listdir(directory) if f.endswith('.h5')]
    ids = [os.path.splitext(f)[0] for f in filenames]

    worker = quantile_worker(modeldir, **kwargs)
    if processes > 1:
        pool = Pool(processes=processes)
        dfs = pool.map(worker, ids)
    else:
        dfs = map(worker, ids)

    df = pd.concat(dfs)
    if filename is None:
        filename = os.path.join(directory,'..','{}_summary.h5'.format(directory))
    df = df.sort_index()
    df.to_hdf(filename, 'df')

    print('Summary dataframe written to {}'.format(filename))
    return df