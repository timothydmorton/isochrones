from __future__ import division,print_function

__author__ = 'Timothy D. Morton <tim.morton@gmail.com>'
"""


"""

import numpy as np
import os,sys,re,os.path
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

from scipy.interpolate import LinearNDInterpolator as interpnd
import numpy.random as rand

try:
    from astropy import constants as const

    #Define useful constants
    G = const.G.cgs.value
    MSUN = const.M_sun.cgs.value
    RSUN = const.R_sun.cgs.value
except ImportError:
    G = 6.67e-11
    MSUN = 1.99e33
    RSUN = 6.96e10
    
import matplotlib.pyplot as plt

try:
    from plotutils.plotutils import setfig
except ImportError:
    setfig = None

from .extinction import EXTINCTION

class Isochrone(object):
    """
    Generic isochrone class. Everything is a function of mass, log(age), Fe/H.

    Can be instantiated directly, but will typically be used with a pre-defined
    subclass, such as :class:`dartmouth.Dartmouth_Isochrone`.  All parameters
    must be array-like objects of the same length, with the exception of ``mags``,
    which is a dictionary of such array-like objects.

    :param m_ini:
        Array of initial mass values [msun].
    :type m_ini: array-like

    :param age: 
        log10(age) [yr]

    :param feh:
        Metallicity [dex]

    :param m_act: 
        Actual mass; same as m_ini if mass loss not implemented [msun]

    :param logL:
        log10(luminosity) [solar units]

    :param Teff:
        Effective temperature [K]

    :param logg:
        log10(surface gravity) [cgs]

    :param mags: 
        Dictionary of absolute magnitudes in different bands
    :type mags: ``dict``
        
    :param tri:
        Triangulation object used
        to initialize the interpolation functions.
        If pre-computed triangulation not provided, then the constructor
        will calculate one.  This might take several minutes, so be patient.
        Much better to use pre-computed ones, as provided in, e.g.,
        :class:`dartmouth.Dartmouth_Isochrone`.
    :type tri: :class:`scipy.spatial.qhull.Delaunay`, optional
        
    """
    def __init__(self,m_ini,age,feh,m_act,logL,Teff,logg,mags,tri=None):
        """Warning: if tri object not provided, this will be very slow to be created.
        """

        self.minage = age.min()
        self.maxage = age.max()
        self.minmass = m_act.min()
        self.maxmass = m_act.max()
        self.minfeh = feh.min()
        self.maxfeh = feh.max()
        

        L = 10**logL

        if tri is None:
            points = np.zeros((len(m_ini),2))
            points[:,0] = m_ini
            points[:,1] = age
            self.mass = interpnd(points,m_act)
            self.tri = self.mass.tri
        else:
            self.tri = tri
            self.mass = interpnd(self.tri,m_act)

        self.logL = interpnd(self.tri,logL)
        self.logg = interpnd(self.tri,logg)
        self.logTeff = interpnd(self.tri,np.log10(Teff))

        def Teff_fn(*pts):
            return 10**self.logTeff(*pts)
        self.Teff = Teff_fn
        
        def R_fn(*pts):
            return np.sqrt(G*self.mass(*pts)*MSUN/10**self.logg(*pts))/RSUN
        self.radius = R_fn

        self.bands = []
        for band in mags.keys():
            self.bands.append(band)

        self.mag = {band:interpnd(self.tri,mags[band]) for band in self.bands}       
        
    def __call__(self, mass, age, feh, 
                 distance=None, AV=0.0,
                 return_df=True, bands=None):
        """
        Returns all properties (or arrays of properties) at given mass, age, feh

        :param mass, age, feh:
            Mass, log(age), metallicity.  Can be float or array_like.

        :param distance:
            Distance in pc.  If passed, then mags will be converted to
            apparent mags based on distance (and ``AV``).

        :param AV:
            V-band extinction (magnitudes).

        :param return_df: (optional)
            If ``True``, return :class:``pandas.DataFrame`` containing all model
            parameters at each input value; if ``False``, return dictionary
            of the same.

        :param bands: (optional)
            List of photometric bands in which to return magnitudes.
            Must be subset of ``self.bands``.  If not set, then will
            default to returning all available bands. 

        :return:
            Either a :class:`pandas.DataFrame` or a dictionary containing
            model values evaluated at input points.
        """
        args = (mass, age, feh)
        Ms = self.mass(*args)
        Rs = self.radius(*args)
        logLs = self.logL(*args)
        loggs = self.logg(*args)
        Teffs = self.Teff(*args)
        if bands is None:
            bands = self.bands
        mags = {band:self.mag[band](*args) for band in bands}
        
        props = {'age':age,'mass':Ms,'radius':Rs,'logL':logLs,
                'logg':loggs,'Teff':Teffs,'mag':mags}        
        if not return_df:
            return props
        else:
            d = {}
            for key in props.keys():
                if key=='mag':
                    for m in props['mag'].keys():
                        mag = props['mag'][m]
                        if distance is not None:
                            dm = 5*np.log10(distance) - 5
                            A = AV*EXTINCTION[m]
                            mag = mag + dm + A
                        d['{}_mag'.format(m)] = mag
                else:
                    d[key] = props[key]
            try:
                df = pd.DataFrame(d)
            except ValueError:
                df = pd.DataFrame(d, index=[0])
            return df

    def agerange(self, m, feh=0.0):
        """
        For a given mass and feh, returns the min and max allowed ages.
        """
        ages = np.arange(self.minage, self.maxage, 0.01)
        rs = self.radius(m, ages, feh)
        w = np.where(np.isfinite(rs))[0]
        return ages[w[0]],ages[w[-1]]

    def evtrack(self,m,feh=0.0,minage=None,maxage=None,dage=0.02,
                return_df=True):
        """
        Returns evolution track for a single initial mass and feh.

        :param m: 
            Initial mass of desired evolution track.

        :param feh: (optional) 
            Metallicity of desired track.  Default = 0.0 (solar)

        :param minage, maxage: (optional)
            Minimum and maximum log(age) of desired track. Will default
            to min and max age of model isochrones. 

        :param dage: (optional)
            Spacing in log(age) at which to evaluate models.  Default = 0.02

        :param return_df: (optional)
            Whether to return a ``DataFrame`` or dicionary.  Default is ``True``.
            

        :return:
            Either a :class:`pandas.DataFrame` or dictionary
            representing the evolution
            track---fixed mass, sampled at chosen range of ages.
        
        """
        if minage is None:
            minage = self.minage
        if maxage is None:
            maxage = self.maxage
        ages = np.arange(minage,maxage,dage)
        Ms = self.mass(m,ages,feh)
        Rs = self.radius(m,ages,feh)
        logLs = self.logL(m,ages,feh)
        loggs = self.logg(m,ages,feh)
        Teffs = self.Teff(m,ages,feh)
        mags = {band:self.mag[band](m,ages,feh) for band in self.bands}

        props = {'age':ages,'mass':Ms,'radius':Rs,'logL':logLs,
                'logg':loggs, 'Teff':Teffs, 'mag':mags}

        if not return_df:
            return props
        else:
            d = {}
            for key in props.keys():
                if key=='mag':
                    for m in props['mag'].keys():
                        d['{}_mag'.format(m)] = props['mag'][m]
                else:
                    d[key] = props[key]
            try:
                df = pd.DataFrame(d)
            except ValueError:
                df = pd.DataFrame(d, index=[0])
            return df

            
    def isochrone(self,age,feh=0.0,minm=None,maxm=None,dm=0.02,
                  return_df=True):
        """
        Returns stellar models at constant age and feh, for a range of masses

        :param age: 
            log10(age) of desired isochrone.

        :param feh: (optional)
            Metallicity of desired isochrone (default = 0.0)

        :param minm, maxm: (optional)
            Mass range of desired isochrone (will default to max and min available)

        :param dm: (optional)
            Spacing in mass of desired isochrone.  Default = 0.02 Msun.

        :param return_df: (optional)
            Whether to return a :class:``pandas.DataFrame`` or dictionary.  Default is ``True``.
        
        :return:
            :class:`pandas.DataFrame` or dictionary containing results.
        
        """
        if minm is None:
            minm = self.minmass
        if maxm is None:
            maxm = self.maxmass
        ms = np.arange(minm,maxm,dm)
        ages = np.ones(ms.shape)*age

        Ms = self.mass(ms,ages,feh)
        Rs = self.radius(ms,ages,feh)
        logLs = self.logL(ms,ages,feh)
        loggs = self.logg(ms,ages,feh)
        Teffs = self.Teff(ms,ages,feh)
        mags = {band:self.mag[band](ms,ages,feh) for band in self.bands}
        #for band in self.bands:
        #    mags[band] = self.mag[band](ms,ages)

        props = {'M':Ms,'R':Rs,'logL':logLs,'logg':loggs,
                'Teff':Teffs,'mag':mags}        
        
        if not return_df:
            return props
        else:
            d = {}
            for key in props.keys():
                if key=='mag':
                    for m in props['mag'].keys():
                        d['{}_mag'.format(m)] = props['mag'][m]
                else:
                    d[key] = props[key]
            try:
                df = pd.DataFrame(d)
            except ValueError:
                df = pd.DataFrame(d, index=[0])
            return df
       
    def random_points(self,n,minmass=None,maxmass=None,
                      minage=None,maxage=None,
                      minfeh=None,maxfeh=None):
        """
        Returns n random mass, age, feh points, none of which are out of range.

        :param n:
            Number of desired points.

        :param minmass, maxmass: (optional)
            Desired allowed range.  Default is mass range of ``self``.

        :param minage, maxage: (optional)
            Desired allowed range.  Default is log10(age) range of
            ``self``.

        :param minfehs, maxfeh: (optional)
            Desired allowed range.  Default is feh range of ``self``.
                        
        :return:
            :class:`np.ndarray` arrays of randomly selected mass, log10(age),
            and feh values
            within allowed ranges.  Used, e.g., to initialize random walkers for
            :class:`StarModel` fits.
        """
        if minmass is None:
            minmass = self.minmass
        if maxmass is None:
            maxmass = self.maxmass
        if minage is None:
            minage = self.minage
        if maxage is None:
            maxage = self.maxage
        if minfeh is None:
            minfeh = self.minfeh
        if maxfeh is None:
            maxfeh = self.maxfeh

        ms = rand.uniform(minmass,maxmass,size=n)
        ages = rand.uniform(minage,maxage,size=n)
        fehs = rand.uniform(minage,maxage,size=n)

        Rs = self.radius(ms,ages,fehs)
        bad = np.isnan(Rs)
        nbad = bad.sum()
        while nbad > 0:
            ms[bad] = rand.uniform(minmass,maxmass,size=nbad)
            ages[bad] = rand.uniform(minage,maxage,size=nbad)
            fehs[bad] = rand.uniform(minfeh,maxfeh,size=nbad)
            Rs = self.radius(ms,ages,fehs)
            bad = np.isnan(Rs)
            nbad = bad.sum()
        return ms,ages,fehs
