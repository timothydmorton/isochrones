from __future__ import division,print_function

import numpy as np
import re,os,os.path
import pandas as pd

FOLDER = os.path.expanduser('~/.isochrones/dartmouth')

def fehstr(feh,minfeh=-1.0,maxfeh=0.5):
        if feh < minfeh:
            return '%.1f' % minfeh
        elif feh > maxfeh:
            return '%.1f' % maxfeh
        elif (feh > -0.05 and feh < 0):
            return '0.0'
        else:
            return '%.1f' % feh            

def writeall_h5(fehs=np.arange(-1,0.51,0.1)):
    store = pd.HDFStore('%s/dartmouth.h5' % FOLDER)
    for feh in fehs:
        name = fehstr(feh)
        print(name)
        store[name] = dartmouth_to_df(feh)
    store.close()

def dartmouth_all_df(fehs=np.arange(-1,0.51,0.1),savefile=None):
    store = pd.HDFStore('%s/dartmouth.h5' % FOLDER)
    df = pd.DataFrame()
    for feh in fehs:
        newdf = store[fehstr(feh)].copy()
        newdf['feh'] = np.ones(len(newdf))*feh
        df = df.append(newdf)
    store.close()
    if savefile is not None:
        df.to_hdf(savefile)
    return df

def dartmouth_to_df(feh):
    filename_2mass = '%s/dartmouth_%s_2massKp.iso' % (FOLDER,fehstr(feh,-1.0,0.5))
    filename_ugriz = '%s/dartmouth_%s_ugriz.iso' % (FOLDER,fehstr(feh,-1.0,0.5))    

    bands_ugriz = ['u','g','r','i','z']
    darnames_ugriz = ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']

    bands_2mass = ['U','B','V','R','I','J','H','Ks','Kp','D51']
    darnames_2mass = bands_2mass

    rec = np.recfromtxt(filename_2mass,skiprows=8,names=True)
    rec2 = np.recfromtxt(filename_ugriz,skiprows=8,names=True,usecols=(5,6,7,8,9))
    df = pd.DataFrame(rec)
    df2 = pd.DataFrame(rec2)
    for b,d in zip(bands_ugriz,darnames_ugriz):
        df[b] = df2[d]

    n = len(df)
    ages = np.zeros(n)
    curage = 0
    i=0
    for line in open(filename_2mass):
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
    
    columns = {darnames_2mass[i]:bands_2mass[i] for i in range(len(bands_2mass))}
    columns.update({'MMo':'M','LogTeff':'logTeff',
                    'LogG':'logg','LogLLo':'logL'})
    df.rename(columns=columns,inplace=True)
    return df
