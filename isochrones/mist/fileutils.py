import os, re
import numpy as np
import pandas as pd

DIR = os.path.join(os.getenv('ISOCHRONES', os.path.expanduser('~/.isochrones')), 'mist',
                  'MIST_v1.0_SDSS')

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

def df_all():
    filenames = os.listdir(DIR)
    df = pd.concat([to_df(os.path.join(DIR, f)) for f in filenames if re.search('.cmd$', f)])
    df = df.sort_values(by=['feh','log10_isochrone_age_yr','initial_mass'])
    df.index = [df.feh, df.log10_isochrone_age_yr]
    return df

