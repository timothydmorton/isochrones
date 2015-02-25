#!/usr/bin/env python
from __future__ import print_function, division

import os, os.path, glob, re
import sys

import pandas as pd
import numpy as np

import logging

DARTMOUTH_ISOCHRONE_DIR = '../isochrones/data/isochrones'

FEHS = [-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,
        -1.0, -0.5, 0.0, 0.2,0.3, 0.5]

def df_all(**kwargs):

    df = pd.DataFrame()
    for pattern in file_patterns(**kwargs):
        df = pd.concat((df,build_df(pattern)))
    return df
        
def file_patterns(fehs=FEHS, afe=0.0, y=None):
    """afe in [-0.2 to 0.8]
    """
    if afe < 0:
        pattern_end = 'afem{:.0f}'.format(abs(afe*10))
    else:
        pattern_end = 'afep{:.0f}'.format(afe*10)
    if y is not None:
        pattern_end += 'y{:.0f}'.format(y)

    patterns = []
    for feh in fehs:
        pattern = 'feh'
        if feh < 0:
            pattern += 'm'
        else:
            pattern += 'p'
        pattern += '{:02.0f}'.format(abs(feh*10))
        pattern += pattern_end
        patterns.append(pattern)
    return patterns

def build_df(pattern):
    """
    Groups all files matching pattern into a single DataFrame

    pattern: e.g. fehp00afep0
    """

    m = re.search('feh([pm])(\d+)afe',pattern)
    if m:
        feh = int(m.group(2))/10
        if m.group(1)=='m':
            feh *= -1
    
    
    files = glob.glob(os.path.join(DARTMOUTH_ISOCHRONE_DIR,
                                   '*',pattern+'.*'))
    files_2 = glob.glob(os.path.join(DARTMOUTH_ISOCHRONE_DIR,
                                   '*',pattern+'*_2'))

    files = list(set(files) - set(files_2))

    if len(files)==0:
        print('no files matching {}')
        return pd.DataFrame()
        
    ages = []
    ages_2 = []
    eeps = []
    eeps_2 = []
    for line in open(files[0]):
        m = re.match('#AGE=\s*(\d+\.\d+)\s+EEPS=(\d+)',line)
        if m:
            ages.append(float(m.group(1)))
            eeps.append(int(m.group(2)))
            
    for line in open(files_2[0]):
        m = re.match('#AGE=\s*(\d+\.\d+)\s+EEPS=(\d+)',line)
        if m:
            ages_2.append(float(m.group(1)))
            eeps_2.append(int(m.group(2)))


    first = True
    for f in files:

        #get column names
        for line in open(f):
            m = re.search('#EEP',line)
            if m:
                colnames = line.split()
                break
            
        df = pd.read_table(f, comment='#', delim_whitespace=True,
                           skipinitialspace=True,
                           names=colnames)
        if first:
            df_all = df
            first = False
        else:
            for col in df.columns:
                if col not in ['EEP','M/Mo','LogTeff',
                               'LogG','LogL/Lo']:
                    df_all[col] = df[col]
    
    df_all['feh'] = np.ones(len(df_all))*feh
    ages_all = None
    for age, n in zip(ages,eeps):
        if ages_all is None:
            ages_all = np.ones(n)*age
        else:
            ages_all = np.concatenate((ages_all, np.ones(n)*age))
    df_all['age'] = ages_all
    
    #build second DF
           
    first = True
    for f in files_2:
        print(f)

        #get column names
        for line in open(f):
            m = re.search('#EEP',line)
            if m:
                colnames = line.split()
                break

        df = pd.read_table(f, comment='#', delim_whitespace=True,
                           names=colnames)
        if first:
            df_all_2 = df
            first = False
        else:
            for col in df.columns:
                if col not in ['EEP','M/Mo','LogTeff',
                               'LogG','LogL/Lo']:
                    df_all_2[col] = df[col]
    
    df_all_2['feh'] = np.ones_like(df_all_2['#EEP'])*feh
    ages_all = None
    for age, n in zip(ages_2,eeps_2):
        if ages_all is None:
            ages_all = np.ones(n)*age
        else:
            ages_all = np.concatenate((ages_all, np.ones(n)*age))
    df_all_2['age'] = ages_all

    return pd.concat((df_all_2, df_all))
        
