from __future__ import division,print_function
import os.path
import numpy as np

import pandas as pd

def fehstr(feh,minfeh=-2.0,maxfeh=0.2):
        if feh < minfeh:
            return '%.1f' % minfeh
        elif feh > maxfeh:
            return '%.1f' % maxfeh
        elif (feh > -0.05 and feh < 0):
            return '0.0'
        else:
            return '%.1f' % feh            


def padova_to_df(feh):
    filename = DATAFOLDER + '/stars/padova_%s.dat' % fehstr(feh,-2,0.2)    
    age,m_ini,m_act,logL,logT,logg,mbol,kep,g,r,i,z,dd051,J,H,K = \
            loadtxt(filename,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),unpack=True)
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
    
