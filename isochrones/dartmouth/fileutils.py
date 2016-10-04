import os, re, glob
import numpy as np
import pandas as pd

from ..config import ISOCHRONES

DATADIR = os.path.join(ISOCHRONES,'dartmouth')

def get_filenames(phot, afe='afep0', y=''):
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
    return os.path.join(DATADIR,'{}{}.h5'.format(phot, afe, y))

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