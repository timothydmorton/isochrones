import os, re
import tarfile
import logging

import numpy as np
import pandas as pd


from ..config import ISOCHRONES
DATADIR = os.path.join(ISOCHRONES,'mist')


def get_dir(phot, version='1.0'):
    return os.path.join(DATADIR, 'MIST_v{}_{}'.format(version, phot))

def get_feh(filename):
    m = re.search('feh_([mp])([0-9]\.[0-9]{2})_afe', filename)
    if m:
        sign = 1 if m.group(1)=='p' else -1
        return float(m.group(2)) * sign

def to_df(filename):
    with open(filename, 'r') as fin:
        while True:
            line = fin.readline()
            if re.match('# EEP', line):
                column_names = line[1:].split()
                break
    feh = get_feh(filename)
    df = pd.read_table(filename, comment='#', delim_whitespace=True,
                         skip_blank_lines=True, names=column_names)
    df['feh'] = feh
    return df

def extract_master_tarball():
    """Unpack tarball of tarballs
    """
    with tarfile.open(os.path.join(ISOCHRONES, 'mist.tgz')) as tar:
        logging.info('Extracting mist.tgz...')
        tar.extractall(ISOCHRONES)

def phot_tarball_file(phot, version='1.0'):
    return os.path.join(DATADIR, 'MIST_v{}_{}.tar.gz'.format(version, phot))

def extract_phot_tarball(phot, version='1.0'):
    phot_tarball = phot_tarball_file(phot)
    with tarfile.open(phot_tarball) as tar:
        logging.info('Extracting {}...'.format(phot_tarball))
        tar.extractall(DATADIR)

def df_all(phot, version='1.0'):
    d = get_dir(phot, version)
    if not os.path.exists(d):
        extract_phot_tarball(phot, version)
    filenames = os.listdir(d)
    df = pd.concat([to_df(os.path.join(d, f)) for f in filenames if re.search('.cmd$', f)])
    df = df.sort_values(by=['feh','log10_isochrone_age_yr','initial_mass'])
    df.index = [df.feh, df.log10_isochrone_age_yr]
    return df

def hdf_filename(phot, version='1.0'):
    return os.path.join(DATADIR,'MIST_v{}_{}.h5'.format(version, phot))

def get_hdf(phot, version='1.0'):
    h5file = hdf_filename(phot, version)
    try:
        df = pd.read_hdf(h5file, 'df')
    except:
        df = write_hdf(phot, version)
    return df

def write_hdf(phot, version='1.0'):
    df = df_all(phot, version=version)   
    h5file = hdf_filename(phot, version=version)
    df.to_hdf(h5file,'df')
    print('{} written.'.format(h5file))
    return df
