from __future__ import print_function, division
import os
import logging

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from itertools import product
import six

from . import extcurve
from ..config import ISOCHRONES

from scipy.interpolate import RegularGridInterpolator
from isochrones.interp import interp_value_extinction, interp_values_extinction

import sys
# sys.path.append('/Users/tdm/repositories/pystellibs')

import pyphot
filter_lib = pyphot.get_library()

from pystellibs import BaSeL, Kurucz

def get_stellargrid(models):
    if models=='kurucz':
        return Kurucz()
    elif models=='basel':
        return BaSeL()
    elif type(models)==type(type):
        return models()
    else:
        return models

def get_filter(b):
    """ Returns pyphot filter given shorthand name
    """
    filtname = None
    if b in ['g','r','i','z']:
        filtname = u'SDSS_{}'.format(b)
    elif b in ['J','H','Ks']:
        filtname = u'2MASS_{}'.format(b)
    elif b=='K':
        filtname = u'2MASS_Ks'
    elif b=='G':
        filtname = u'Gaia_G'
    elif b in ['W1','W2','W3','W4']:
        filtname = u'WISE_RSR_{}'.format(b)
    elif b in ['U','B','V']:
        filtname = u'GROUND_JOHNSON_{}'.format(b)
    else:
        filtname = six.text_type(b)

    try:
        return filter_lib[filtname]
    except UnicodeDecodeError:
        logging.error('Cannot decode band name: {}'.format(filtname))
        raise 

class ModelSpectrumGrid(object):
    def __init__(self, models=Kurucz):
        self.models = get_stellargrid(models)
        
        self._Nlogg = None
        self._NlogT = None
        self._Nfeh = None
        self._logg_grid = None
        self._logT_grid = None
        self._feh_grid = None
        
        self._prop_df = None
        self._spectrum = None
        self._lam = None
                
            
    @property
    def Nlogg(self):
        if self._Nlogg is None:
            self._Nlogg = len(self.logg_grid)
        return self._Nlogg
            
    @property
    def NlogT(self):
        if self._NlogT is None:
            self._NlogT = len(self.logT_grid)
        return self._NlogT

    @property
    def Nfeh(self):
        if self._Nfeh is None:
            self._Nfeh = len(self.feh_grid)
        return self._Nfeh
            
    @property
    def logg_grid(self):
        if self._logg_grid is None:
            self.prop_df
        return self._logg_grid

    @property
    def logT_grid(self):
        if self._logT_grid is None:
            self.prop_df
        return self._logT_grid

    @property
    def feh_grid(self):
        if self._feh_grid is None:
            self.prop_df
        return self._feh_grid

    @property
    def prop_df(self):
        if self._prop_df is None:
            loggs = np.sort(np.unique(self.models.grid['logg']))
            logTs = np.sort(np.unique(self.models.grid['logT']))
            Zs = np.sort(np.unique(self.models.grid['Z']))
            fehs = np.log10(Zs/0.014)
            self._logg_grid = loggs
            self._logT_grid = logTs
            self._feh_grid = fehs
            N = len(loggs)*len(logTs)*len(Zs)

            d = {'logg':np.zeros(N), 'logT':np.zeros(N), 'Z':np.zeros(N), 'logL':np.zeros(N)}
            for i, (g,T,Z) in enumerate(product(loggs, logTs, Zs)):    
                d['logg'][i] = g
                d['logT'][i] = T
                d['Z'][i] = Z

            d['feh'] = np.log10(d['Z']/0.014)
            self._prop_df = pd.DataFrame(d)
        return self._prop_df
        
    def _generate_grid(self):
        lam, spec = self.models.generate_individual_spectra(self.prop_df)
        self._spectrum = spec
        self._lam = lam
        
    @property
    def spectrum(self):
        if self._spectrum is None:
            self._generate_grid()
        return self._spectrum
    
    @property
    def lam(self):
        if self._lam is None:
            self._generate_grid()
        return self._lam
            
    def get_Agrid(self, band, AV_grid, x=0., savefile=None):
        filt = get_filter(band)
        ext = extcurve(x)
        lam = self.lam
        spec = self.spectrum
        flux_clean = filt.get_flux(lam, spec)
        shape = (self.Nlogg, self.NlogT, self.Nfeh, len(AV_grid))
        dmag = np.zeros(shape)
        for i,AV in enumerate(AV_grid):
            spec_atten = spec*np.exp(-0.4*AV*ext(lam.to('angstrom').magnitude))
            flux_atten = filt.get_flux(lam, spec_atten)

            # Why does this only make sense with log instead of log10????
            dmag[:,:,:,i] = -2.5*np.log(flux_atten / flux_clean).reshape((shape[:3]))

        return ExtinctionGrid(band, dmag, self.logg_grid, 
                              self.logT_grid, self.feh_grid, AV_grid)
    

class ExtinctionGrid(object):
    def __init__(self, band, Agrid, logg, logT, feh, AV, use_scipy=False):
        assert Agrid.shape == (len(logg), len(logT), len(feh), len(AV))
        self.band = band
        self.Agrid = Agrid.astype(float)
        self.logg = logg.astype(float)
        self.logT = logT.astype(float)
        self.feh = feh.astype(float)
        self.AV = AV.astype(float)

        self.logg_norm = self.logg[-1] - self.logg[0]
        self.logT_norm = self.logT[-1] - self.logT[0]
        self.feh_norm = self.feh[-1] - self.feh[0]
        self.AV_norm = self.AV[-1] - self.AV[0]

        self.use_scipy = False
        self._scipy_func = None

        self._flat_points = None
        self._flat_Agrid = None
        
        self._kdtree = None
        self._kdtree_pts = None
        self._kdtree_vals = None

    @property
    def flat_points(self):
        if self._flat_points is None:
            self._flat_points = np.array([[g, T, f, A] for 
                                            g, T, f, A in product(self.logg,
                                                              self.logT,
                                                              self.feh,
                                                              self.AV)])
        return self._flat_points

    @property
    def flat_Agrid(self):
        if self._flat_Agrid is None:
            self._flat_Agrid = self.Agrid.ravel()
        return self._flat_Agrid
    
    @property    
    def kdtree(self):
        if self._kdtree is None:
            pts = self.flat_points
            pts[:, 0] /= self.logg_norm
            pts[:, 1] /= self.logT_norm
            pts[:, 2] /= self.feh_norm
            pts[:, 3] /= self.AV_norm
            vals = self.flat_Agrid
            ok = np.isfinite(vals)
            self._kdtree_pts = pts[ok, :]
            self._kdtree_vals = vals[ok]
            self._kdtree = KDTree(self._kdtree_pts, leaf_size=2)
        return self._kdtree

    def _kdtree_interp(self, pars):
        pars = np.atleast_2d(pars)
        pars[:, 0] /= self.logg_norm
        pars[:, 1] /= self.logT_norm
        pars[:, 2] /= self.feh_norm
        pars[:, 3] /= self.AV_norm
        d, i = self.kdtree.query(pars)
        return self._kdtree_vals[i]

    def _build_scipy_func(self):    
        points = (self.logg / self.logg_norm, 
                  self.logT / self.logT_norm, 
                  self.feh / self.feh_norm, 
                  self.AV / self.AV_norm)
        vals = self.Agrid
        self._scipy_func = RegularGridInterpolator(points, vals)

    def _scipy_interp(self, pars):
        g, T, f, A = pars
        g = g / self.logg_norm
        T = T / self.logT_norm
        f = f / self.feh_norm
        A = A / self.AV_norm
        if self._scipy_func is None:
            self._build_scipy_func()
        return self._scipy_func([g, T, f, A])
    
    def _custom_interp(self, pars):
        g, T, f, A = pars
        g = g / self.logg_norm
        T = T / self.logT_norm
        f = f / self.feh_norm
        A = A / self.AV_norm
        if np.size(g)==1 and np.size(T)==1 and np.size(f)==1 and np.size(A)==1:
            try:
                return interp_value_extinction(g, T, f, A, self.Agrid, 
                                        self.logg / self.logg_norm,
                                        self.logT / self.logT_norm, 
                                        self.feh / self.feh_norm, 
                                        self.AV / self.AV_norm)
            except ZeroDivisionError:
                return self._kdtree_interp(pars)
        else:
            # return self._kdtree_interp(pars)
            b = np.broadcast(g, T, f, A)
            g = np.resize(g, b.shape).astype(float)
            T = np.resize(T, b.shape).astype(float)
            f = np.resize(f, b.shape).astype(float)
            A = np.resize(A, b.shape).astype(float)

            return interp_values_extinction(g, T, f, A, self.Agrid, self.logg,
                                          self.logT, self.feh, self.AV)


    def __call__(self, *args, **kwargs):
        use_scipy = self.use_scipy or ('scipy' in kwargs and kwargs['scipy'])
        if use_scipy:
            if self._scipy_func is None:
                self._build_scipy_func()
            return self._scipy_interp(*args)
        else:
            return self._custom_interp(*args)

    def save(self, filename):
        """Saves as a numpy save file
        """
        np.savez(filename, Agrid=self.Agrid, logg=self.logg, 
                 logT=self.logT, feh=self.feh, AV=self.AV,
                 band=np.array([self.band]))

    @classmethod
    def load(cls, filename):
        d = np.load(filename)
        return cls(d['band'][0], d['Agrid'], 
                    d['logg'], d['logT'], d['feh'], d['AV'])


def extinction_grid_filename(band, x=0., models='kurucz'):
    extinction_dir = os.path.join(ISOCHRONES, 'extinction')
    if not os.path.exists(extinction_dir):
        os.makedirs(extinction_dir)
    filename = os.path.join(extinction_dir, '{}_{}_{:.2f}.npz'.format(band, models, x))
    return filename    

def get_extinction_grid(band, AVs=np.concatenate(([0], np.logspace(-1,1,12))),
                            x=0., models='kurucz', overwrite=False):
    filename = extinction_grid_filename(band, x=x, models=models)
    try:
        if overwrite:
            raise IOError
        Agrid = ExtinctionGrid.load(filename)
    except (IOError, KeyError):
        write_extinction_grid(band, AVs=AVs, x=x, models=models)
        grid = ModelSpectrumGrid(models)
        Agrid = grid.get_Agrid(band, AVs, x=x)
    return Agrid

def write_extinction_grid(band, AVs=np.concatenate(([0], np.logspace(-1,1,12))),
                            x=0., models='kurucz'):
    grid = ModelSpectrumGrid(models)
    Agrid = grid.get_Agrid(band, AVs, x=x)
    filename = extinction_grid_filename(band, x=x, models=models)
    Agrid.save(filename)
    logging.info('Extinction grid saved to {}.'.format(filename))

