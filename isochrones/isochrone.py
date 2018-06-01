from .config import on_rtd

import os, re, sys
import warnings
import logging

if not on_rtd:
    import pandas as pd
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator as interpnd
    import numpy.random as rand
    import matplotlib.pyplot as plt

    from astropy import constants as const

    #Define useful constants
    G = const.G.cgs.value
    MSUN = const.M_sun.cgs.value
    RSUN = const.R_sun.cgs.value

    from .extinction import EXTINCTION, LAMBDA_EFF, extcurve, extcurve_0
    from .interp import interp_value, interp_values

else:
    G = 6.67e-11
    MSUN = 1.99e33
    RSUN = 6.96e10

from .config import ISOCHRONES
from .grid import ModelGrid

def get_ichrone(models, bands=None, default=False, **kwargs):
    """Gets Isochrone Object by name, or type, with the right bands

    If `default` is `True`, then will set bands
    to be the union of bands and default_bands
    """
    if isinstance(models, Isochrone):
        return models

    def actual(bands, ictype):
        if bands is None:
            return list(ictype.default_bands)
        elif default:
            return list(set(bands).union(set(ictype.default_bands)))
        else:
            return bands

    if type(models) is type(type):
        ichrone = models(actual(bands, models))
    elif models=='dartmouth':
        from isochrones.dartmouth import Dartmouth_Isochrone
        ichrone = Dartmouth_Isochrone(bands=actual(bands, Dartmouth_Isochrone), **kwargs)
    elif models=='dartmouthfast':
        from isochrones.dartmouth import Dartmouth_FastIsochrone
        ichrone = Dartmouth_FastIsochrone(bands=actual(bands, Dartmouth_FastIsochrone), **kwargs)
    elif models=='mist':
        from isochrones.mist import MIST_Isochrone
        ichrone = MIST_Isochrone(bands=actual(bands, MIST_Isochrone), **kwargs)
    elif models=='padova':
        from isochrones.padova import Padova_Isochrone
        ichrone = Padova_Isochrone(bands=actual(bands, Padova_Isochrone), **kwargs)
    elif models=='basti':
        from isochrones.basti import Basti_Isochrone
        ichrone = Basti_Isochrone(bands=actual(bands, Basti_Isochrone), **kwargs)
    else:
        raise ValueError('Unknown stellar models: {}'.format(models))
    return ichrone


class Isochrone(object):
    """
    Basic isochrone class. Everything is a function of mass, log(age), Fe/H.

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

    :param minage,maxage:
        If desired, a minimum or maximum age can be manually entered.

    """
    def __init__(self,m_ini,age,feh,m_act,logL,Teff,logg,mags,tri=None,
                 minage=None, maxage=None, ext_table=False):
        """Warning: if tri object not provided, this will be very slow to be created.
        """

        self.minage = age.min()
        self.maxage = age.max()
        self.minmass = m_act.min()
        self.maxmass = m_act.max()
        self.minfeh = feh.min()
        self.maxfeh = feh.max()

        self.ext_table = ext_table

        if minage is not None:
            logging.warning("minage and maxage keywords are deprecated." + \
                          "Use instead the .set_bounds(age=(lo, hi)) attribute of StarModel.")
            self.minage = minage
        if maxage is not None:
            logging.warning("minage and maxage keywords are deprecated." + \
                          "Use instead the .set_bounds(age=(lo, hi)) attribute of StarModel.")
            self.maxage = maxage

        L = 10**logL

        if tri is None:
            points = np.zeros((len(m_ini),3))
            points[:,0] = m_ini
            points[:,1] = age
            points[:,2] = feh
            fn = interpnd(points,m_act)
            self.tri = fn.tri
        else:
            self.tri = tri
            self.mass = interpnd(self.tri,m_act)

        self._data = {'mass':m_act,
                    'logL':logL,
                    'logg':logg,
                    'logTeff':np.log10(Teff),
                    'mags':mags}
        self._props = ['mass', 'logL', 'logg', 'logTeff']

        self.bands = list(mags.keys())

        self._mag = {band:interpnd(self.tri,mags[band]) for band in self.bands}

        self.mag = {b : self._mag_fn(b) for b in self.bands}

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['mag'] # This can't be pickled
        return odict

    def __setstate__(self, odict):
        self.__dict__ = odict
        self.__dict__['mag'] = {b : self._mag_fn(b) for b in self.bands}

    def _prop(self, prop, *args):
        if prop not in self._props:
            raise ValueError('Cannot call this function with {}.'.format(prop))
        attr = '_{}'.format(prop)
        if not hasattr(self, attr):
            setattr(self, attr, interpnd(self.tri, self._data[prop]))
        fn = getattr(self, attr)
        return fn(*args)

    def mass(self, *args):
        return self._prop('mass', *args)

    def logL(self, *args):
        return self._prop('logL', *args)

    def logg(self, *args):
        return self._prop('logg', *args)

    def logTeff(self, *args):
        return self._prop('logTeff', *args)

    def radius(self, *args):
        return np.sqrt(G*self.mass(*args)*MSUN/10**self.logg(*args))/RSUN

    def Teff(self, *args):
        return 10**self.logTeff(*args)

    def density(self, *args):
        """ Mean density in g/cc
        """
        M = self.mass(*args) * MSUN
        V = 4./3 * np.pi * (self.radius(*args) * RSUN)**3
        return  M/V

    def delta_nu(self, *args):
        """Returns asteroseismic delta_nu in uHz

        reference: https://arxiv.org/pdf/1312.3853v1.pdf, Eq (2)
        """
        return 134.88 * np.sqrt(self.mass(*args) / self.radius(*args)**3)

    def nu_max(self, *args):
        """Returns asteroseismic nu_max in uHz

        reference: https://arxiv.org/pdf/1312.3853v1.pdf, Eq (3)
        """
        return 3120.* (self.mass(*args) /
                        (self.radius(*args)**2 * np.sqrt(self.Teff(*args)/5777.)))

    def _mag_fn(self, band):
        def fn(mass, age, feh, distance=10, AV=0.0, x_ext=0., ext_table=self.ext_table):
            if x_ext==0.:
                ext = extcurve_0
            else:
                ext = extcurve(x_ext)
            if ext_table:
                A = AV*EXTINCTION[band]
            else:
                A = AV*ext(LAMBDA_EFF[band])
            dm = 5*np.log10(distance) - 5
            return self._mag[band](mass, age, feh) + dm + A
        return fn


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
        # Broadcast inputs to the same shape
        mass, age, feh = [np.array(a) for a in np.broadcast_arrays(mass, age, feh)]
        args = (mass, age, feh)
        Ms = self.mass(*args)*1
        Rs = self.radius(*args)*1
        logLs = self.logL(*args)*1
        loggs = self.logg(*args)*1
        Teffs = self.Teff(*args)*1
        if bands is None:
            bands = self.bands
        if distance is not None:
            args += (distance, AV)
        mags = {band:1*self.mag[band](*args) for band in bands}
        # if distance is not None:
        #     dm = 5*np.log10(distance) - 5
        #     for band in mags:
        #         A = AV*EXTINCTION[band]
        #         mags[band] = mags[band] + dm + A


        props = {'age':age,'mass':Ms,'radius':Rs,'logL':logLs,
                'logg':loggs,'Teff':Teffs,'mag':mags}

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
            df = pd.DataFrame(d)
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
                  return_df=True,distance=None,AV=0.0):
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

        :param distance:
            Distance in pc.  If passed, then mags will be converted to
            apparent mags based on distance (and ``AV``).

        :param AV:
            V-band extinction (magnitudes).

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
        if distance is not None:
            dm = 5*np.log10(distance) - 5
            for band in mags:
                A = AV*EXTINCTION[band]
                mags[band] = mags[band] + dm + A

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

        .. todo::

            Should change this to drawing from priors!  Current implementation
            is a bit outdated.
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


class MagFunction(object):
    def __init__(self, ic, band, icol):
        self.ic = ic
        self.band = band
        self.icol = icol
        self.x_ext = ic.x_ext
        self.ext_table = ic.ext_table

        if self.x_ext==0.:
            ext = extcurve_0
        else:
            ext = extcurve(x_ext)
        if self.ext_table:
            self.AAV = EXTINCTION[self.band]
        else:
            self.AAV = ext(LAMBDA_EFF[self.band])

    def __call__(self, mass, age, feh, distance=10, AV=0.0, x_ext=None, ext_table=False):
        if x_ext is not None:
            if x_ext==0.:
                ext = extcurve_0
            else:
                ext = extcurve(x_ext)

            if ext_table:
                AAV = EXTINCTION[self.band]
            else:
                AAV = ext(LAMBDA_EFF[self.band])
        else:
            AAV = self.AAV

        A = AV*AAV
        dm = 5*np.log10(distance) - 5
        mag = self.ic.interp_value(mass, age, feh, self.icol)
        return mag + dm + A

class FastIsochrone(Isochrone):
    """Alternative isochrone implementation for large grids, faster likelihoods

    This implementation allows faster point interpolations than
    the triangulation method in the base :class:`Isochrone` class;
    however, for simulating large populations (passing arrays of parameters
    at a time), the base :class:`Isochrone` is still faster.  This is
    also the go-to implementation for grids that are too large to make the
    Delaunay method feasible (e.g., significantly over 100,000 points in the grid.)

    However, edge effects might be a bit more problematic here than in the base class,
    meaning fitting very evolved stars might be an issue. Also, don't trust this subclass
    for the :function:`Isochrone.agerange` function, for the same reason.

    Subclasses must set the appropriate attributes, and then things should work.
    """
    name = 'default'
    modelgrid = ModelGrid
    age_col = None
    feh_col = None
    mass_col = None
    loggTeff_col = None
    logg_col = None
    logL_col = None
    default_bands = ('g')

    def __init__(self, bands=None, x_ext=0., ext_table=False, debug=False, **kwargs):
        # df should be indexed by [feh, age]

        if bands is None:
            bands = list(self.default_bands)
        bands = sorted(bands)
        self.bands = bands

        self._df = None
        self.x_ext = 0.
        self.ext_table = ext_table
        self.debug = debug


        self._fehs = None
        self._ages = None
        self._Nfeh = None
        self._Nage = None

        self._minage = None
        self._maxage = None
        self._minmass = None
        self._maxmass = None
        self._minfeh = None
        self._maxfeh = None

        n_common_cols = len(self.modelgrid.get_common_columns(**kwargs))
        self._mag_cols = {b:n_common_cols+i for i,b in enumerate(self.bands)}
        self.mag = {b: MagFunction(self, b, i)
                            for b,i in self._mag_cols.items()}

        #organized array
        self._grid = None
        self._grid_Ns = None

        # kwargs to pass to self.modelgrid
        self.modelgrid_kwargs = kwargs

    def _initialize(self):
        for attr in ['df','Ncols','fehs','ages','Nfeh','Nage',
                     'minage','maxage','minfeh','maxfeh','minmass','maxmass']:
            _ = getattr(self, attr)


    @property
    def df(self):
        if self._df is None:
            self._df = self.modelgrid(self.bands, **self.modelgrid_kwargs).df
        return self._df

    @property
    def Ncols(self):
        return self.df.shape[1]

    @property
    def fehs(self):
        if self._fehs is None:
            self._fehs = self.df.iloc[:, self.feh_col].unique()
        return self._fehs

    @property
    def ages(self):
        if self._ages is None:
            self._ages = self.df.iloc[:, self.age_col].unique()
        return self._ages

    @property
    def Nfeh(self):
        if self._Nfeh is None:
            self._Nfeh = len(self.fehs)
        return self._Nfeh

    @property
    def Nage(self):
        if self._Nage is None:
            self._Nage = len(self.ages)
        return self._Nage

    @property
    def minage(self):
        if self._minage is None:
            self._minage = self.ages.min()
        return self._minage

    @property
    def maxage(self):
        if self._maxage is None:
            self._maxage = self.ages.max()
        return self._maxage

    @property
    def minfeh(self):
        if self._minfeh is None:
            self._minfeh = self.fehs.min()
        return self._minfeh

    @property
    def maxfeh(self):
        if self._maxfeh is None:
            self._maxfeh = self.fehs.max()
        return self._maxfeh

    @property
    def minmass(self):
        if self._minmass is None:
            self._minmass = self.df.iloc[:, self.mass_col].min()
        return self._minmass

    @property
    def maxmass(self):
        if self._maxmass is None:
            self._maxmass = self.df.iloc[:, self.mass_col].max()
        return self._maxmass

    def logTeff(self, mass, age, feh):
        return self.interp_value(mass, age, feh, self.loggTeff_col)

    def logg(self, mass, age, feh):
        return self.interp_value(mass, age, feh, self.logg_col)

    def logL(self, mass, age, feh):
        return self.interp_value(mass, age, feh, self.logL_col)

    def mass(self, *args):
        if np.size(args[0]) > 1:
            return np.array(args[0])
        else:
            return args[0]

    @property
    def grid(self):
        if self._grid is None:
            self._make_grid()
        return self._grid

    @property
    def grid_Ns(self):
        if self._grid_Ns is None:
            self._make_grid()
        return self._grid_Ns

    @property
    def _npz_filename(self):
        keys = list(self.modelgrid_kwargs.keys())
        keys.sort()

        filename = os.path.join(ISOCHRONES, self.name, '{}'.format('-'.join(self.bands)))

        for k in keys:
            filename += '_{}{}'.format(k, self.modelgrid_kwargs[k])

        filename += '.npz'
        return filename

    def _make_grid(self, recalc=False):
        # Read from file if available.
        if os.path.exists(self._npz_filename) and not recalc:
            d = np.load(self._npz_filename)
            self._grid = d['grid']
            self._grid_Ns = d['grid_Ns']
        else:
            df_list = [[self.df.ix[f,a] for f in self.fehs] for a in self.ages]
            lens = np.array([[len(df_list[i][j]) for j in range(self.Nfeh)]
                             for i in range(self.Nage)]).T #just because
            data = np.zeros((self.Nfeh, self.Nage, lens.max(), self.Ncols))

            for i in range(self.Nage):
                for j in range(self.Nfeh):
                    N = lens[j,i]
                    data[j, i, :N, :] = df_list[i][j].values
                    data[j, i, N:, :] = np.nan

            np.savez(self._npz_filename, grid=data, grid_Ns=lens)
            self._grid = data
            self._grid_Ns = lens

    def interp_value(self, mass, age, feh, icol): # 4 is log_g
        if self._ages is None:
            self._initialize()

        try:
            return interp_value(float(mass), float(age), float(feh), icol,
                                self.grid, self.mass_col,
                                self.ages, self.fehs, self.grid_Ns, self.debug)

        except:
            # First, broadcast to common shape.
            b = np.broadcast(mass, age, feh)
            mass = np.resize(mass, b.shape).astype(float)
            age = np.resize(age, b.shape).astype(float)
            feh = np.resize(feh, b.shape).astype(float)

            # Then pass to helper function
            return interp_values(mass, age, feh, icol,
                                self.grid, self.mass_col,
                                self.ages, self.fehs, self.grid_Ns)
