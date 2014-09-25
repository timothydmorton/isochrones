from __future__ import division,print_function

from scipy.interpolate import LinearNDInterpolator as interpnd
import numpy as np
import re,os,os.path
import pandas as pd
import pickle


DATAFOLDER = '{}/stars'.format(os.environ['ASTROUTIL_DATADIR'])

try:
    DF = pd.read_hdf('isochrones/data/padova.h5','df')
except:
    DF = padova_to_df()
    
def fehstr(feh,minfeh=-2.0,maxfeh=0.2):
        if feh < minfeh:
            return '%.1f' % minfeh
        elif feh > maxfeh:
            return '%.1f' % maxfeh
        elif (feh > -0.05 and feh < 0):
            return '0.0'
        else:
            return '%.1f' % feh            


def padova_to_df(save=True,savefile='isochrones/data/padova.h5'):
    alldf = pd.DataFrame()
    
    for feh in np.arange(-2,0.21,0.1):
        filename = DATAFOLDER + '/padova_%s.dat' % fehstr(feh,-2,0.2)    
        age,m_ini,m_act,logL,logT,logg,mbol,kep,g,r,i,z,dd051,J,H,K = \
                np.loadtxt(filename,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),unpack=True)
        df = pd.DataFrame()
        df['age'] = age
        df['M_ini'] = m_ini
        df['M_act'] = m_act
        df['logL'] = logL
        df['logTeff'] = logT
        df['logg'] = logg
        df['bol'] = mbol
        df['Kepler'] = kep
        df['g'] = g
        df['r'] = r
        df['i'] = i
        df['z'] = z
        df['D51'] = dd051
        df['J'] = J
        df['H'] = H
        df['K'] = K
        df['feh'] = feh * np.ones_like(age)
        alldf = alldf.append(df)

    if save:
        alldf.to_hdf(savefile,'df')
        
    return alldf

def write_tri(df=DF,outfile='padova.tri'):
        N = len(df)
        pts = np.zeros((N,3))
        pts[:,0] = np.array(df['M_ini'])
        pts[:,1] = np.array(df['age'])
        pts[:,2] = np.array(df['feh'])
        Jmags = np.array(df['J'])

        Jfn = interpnd(pts,Jmags)

        f = open(outfile,'wb')
        pickle.dump(Jfn.tri,f)
        f.close()
