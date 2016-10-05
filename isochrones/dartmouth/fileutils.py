import os, re, glob
import tarfile
import logging

import numpy as np
import pandas as pd

from ..config import ISOCHRONES
DATADIR = os.path.join(ISOCHRONES,'dartmouth')

def download_grids(record=159426, overwrite=True):
    tarball_path = os.path.join(ISOCHRONES, 'dartmouth.tgz')
    tarball_url = 'https://zenodo.org/record/{}/files/dartmouth.tgz'.format(record)

    tri_path = os.path.join(ISOCHRONES, 'dartmouth.tri')
    tri_url = 'https://zenodo.org/record/{}/files/dartmouth.tri'.format(record)

    from six.moves import urllib
    print('Downloading Dartmouth stellar model data (should happen only once)...')

    paths = [tarball_path, tri_path]
    urls = [tarball_url, tri_url]
    for path, url in zip(paths, urls):
        if os.path.exists(path):
            if overwrite:
                os.remove(path)
            else:
                continue
        urllib.request.urlretrieve(url, path)


def extract_master_tarball():
    """Unpack tarball of tarballs
    """
    with tarfile.open(os.path.join(ISOCHRONES, 'dartmouth.tgz')) as tar:
        logging.info('Extracting dartmouth.tgz...')
        tar.extractall(ISOCHRONES)

def phot_tarball_file(phot):
    return os.path.join(DATADIR, '{}.tgz'.format(phot))

def extract_phot_tarball(phot):
    phot_tarball = phot_tarball_file(phot)
    with tarfile.open(phot_tarball) as tar:
        logging.info('Extracting {}.tgz...'.format(phot))
        tar.extractall(DATADIR)

def get_filenames(phot, afe='afep0', y=''):
    if not os.path.exists(os.path.join(DATADIR, 'isochrones', phot)):
        if not os.path.exists(phot_tarball_file(phot)):
            extract_master_tarball()
        extract_phot_tarball(phot)

    return glob.glob('{3}/isochrones/{0}/*{1}{2}.{0}*'.format(phot,afe,y,DATADIR))

def get_feh(filename):
    m = re.search('feh([mp])(\d+)afe', filename)
    if m:
        sign = 1 if m.group(1)=='p' else -1
        return float(m.group(2))/10 * sign
    
def to_df(filename):
    try:
        rec = np.recfromtxt(filename,skip_header=8,names=True)
    except:
        print('Error reading {}!'.format(filename))
        raise RuntimeError
    df = pd.DataFrame(rec)
    
    n = len(df)
    ages = np.zeros(n)
    curage = 0
    i=0
    for line in open(filename):
        m = re.match('#',line)
        if m:
            m = re.match('#AGE=\s*(\d+\.\d+)\s+',line)
            if m:
                curage=float(m.group(1))
        else:
            if re.search('\d',line):
                ages[i]=curage
                i+=1

    df['age'] = ages
    df['feh'] = get_feh(filename)
    return df

def df_all(phot, afe='afep0', y=''):
    df = pd.concat([to_df(f) for f in get_filenames(phot, afe=afe, y=y)])
    return df.sort_values(by=['age','feh','MMo','EEP'])
    
def hdf_filename(phot, afe='afep0', y=''):
    afe_str = '_{}'.format(afe) if afe!='afep0' else ''
    return os.path.join(DATADIR,'{}{}.h5'.format(phot, afe_str, y))

def get_hdf(phot, afe='afep0', y=''):
    h5file = hdf_filename(phot, afe=afe, y=y)
    try:
        df = pd.read_hdf(h5file, 'df')
    except:
        df = write_hdf(phot, afe=afe, y=y)
    return df

def write_hdf(phot, afe='afep0', y=''):
    df = df_all(phot, afe=afe, y=y)   
    h5file = hdf_filename(phot, afe=afe, y=y)
    df.to_hdf(h5file,'df')
    print('{} written.'.format(h5file))
    return df