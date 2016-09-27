import os
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
from configobj import ConfigObj, Section

class EmptyQueryError(ValueError):
    pass

class VizierCatalog(object):
    def __init__(self, query):
        self.query = query
        self._table = None
        self._query_coords = None
        self._coords = None
        self._empty = False

    def __repr__(self):
        return '{0}({1})'.format(type(self), repr(self.query))

    def __str__(self):
        return '{} Query of {}'.format(self.name, self.query)

    def _run_query(self):
        if self._empty:
            raise EmptyQueryError('{} is empty!'.format(self))            
        try:
            self._table = Vizier.query_region(self.query_coords, radius=self.query.radius,
                                        catalog=self.vizier_name)[0]
        except IndexError:
            self._empty = True
            raise EmptyQueryError('{} returns empty!'.format(self))
        self._coords = SkyCoord(self._table['_RAJ2000'], self._table['_DEJ2000'], unit='deg')
        self._table['PA'] = self.coords.position_angle(self.query_coords).deg
        #self._table['separation'] = self.coords.separation(self.query_coords).arcsec

    @property
    def coords(self):
        if self._coords is None:
            self._run_query()
        return self._coords
    

    @property
    def query_coords(self):
        if self._query_coords is None:
            q = self.query
            dt = (q.epoch - self.epoch)*u.yr
            ra = q.ra*u.deg - dt*q.pmra*u.mas/u.yr
            dec = q.dec*u.deg - dt*q.pmdec*u.mas/u.yr
            self._query_coords = SkyCoord(ra, dec)
        return self._query_coords

    @property
    def table(self):
        if self._table is None:
            self._run_query()
        return self._table
    
    @property
    def df(self):
        return pd.DataFrame(np.array(self.table))
    
    @property
    def closest(self):
        df = self.df.sort_values(by='_r')
        return df.iloc[0]

    @property
    def brightest(self):
        band = self.bands.keys()[0]
        df = self.df.sort_values(by=band)
        return df.iloc[0]
    
    def get_photometry(self, brightest=False,
                    min_unc=0.02, convert=True):
        """Returns dictionary of photometry of closest match

        unless brightest is True, in which case the brightest match.
        """
        if brightest:
            row = self.brightest
        else:
            row = self.closest

        if not hasattr(self, 'conversions'):
            convert = False

        if convert:
            bands = self.conversions 
        else:
            bands = self.bands.keys()

        d = {}
        for b in bands:
            if convert:
                key = b
                mag, dmag = getattr(self, b)(brightest=brightest)
            else:
                key = self.bands[b]
                mag, dmag = row[b], row['e_{}'.format(b)]

            d[key] = mag, max(dmag, min_unc)
        return d        

class TwoMASS(VizierCatalog):
    name = 'twomass'
    vizier_name = '2mass'
    epoch = 2000.
    bands = {'Jmag':'J', 
             'Hmag':'H', 
             'Kmag':'K'}

class Tycho2(VizierCatalog):
    name = 'Tycho2'
    vizier_name = 'tycho2'
    epoch = 2000
    bands = {'BTmag':'BT', 'VTmag':'VT'}
    conversions = ['B','V']

    def V(self, brightest=False):
        """
        http://www.aerith.net/astro/color_conversion.html
        """
        mags = self.get_photometry(brightest=brightest, convert=False)
        VT, dVT = mags['VT']
        BT, dBT = mags['BT']
        if (-0.25 < BT - VT < 2.0):
            (a, b, c, d) = (0.00097, 0.1334, 0.05486, 0.01998)
            V = (VT + a - b * (BT - VT) + c * (BT - VT)**2 - 
                d * (BT - VT)**3)

            dVdVT = 1 + b - 2*c*(BT-VT) + 3*d*(BT-VT)**2
            dVdBT = -b + 2*c*(BT-VT) - 3*d*(BT-VT)**2
            dV = np.sqrt((dVdVT**2 * dVT**2) + (dVdBT**2*dBT**2))

        else:
            raise ValueError('BT-VT outside of range to convert')

        return V, dV

    def BmV(self, brightest=False):
        mags = self.get_photometry(brightest=brightest, convert=False)
        VT, dVT = mags['VT']
        BT, dBT = mags['BT']
        if 0.5 < (BT-VT) < 2.0:
            (e, f, g) = (0.007813, 0.1489, 0.03384)
            BmV = ((BT - VT) - e * (BT - VT) - 
                    f * (BT - VT)**2 + g * (BT - VT)**3)

            dBmVdVT = -1 + e + 2*f*(BT-VT) - 3*g*(BT-VT)**2  
            dBmVdBT = -dBmVdVT

        elif -0.25 < (BT - VT) < 0.5:
            (h, i, j) = (0.006, 0.1069, 0.1459)
            BmV = ((BT - VT) - h - i * (BT - VT) + 
                    j * (BT - VT)**2)

            dBmVdVT = -1 - i - 2*j*(BT-VT)
            dBmVdBT = -dBmVdVT

        else:
            raise ValueError('BT-VT outside of range to convert')

        dBmV = np.sqrt((dBmVdVT**2 * dVT**2) + (dBmVdBT**2*dBT**2))
        return BmV, dBmV

    def B(self, brightest=False):
        BmV, dBmV = self.BmV(brightest=brightest)
        V, dV = self.V(brightest=brightest)

        B = BmV + V
        dB = np.sqrt(dBmV**2 + dV**2)
        return B, dB

class WISE(VizierCatalog):
    name = 'WISE'
    vizier_name = 'allwise'
    epoch = 2000
    bands = {'W1mag':'W1', 'W2mag':'W2', 
             'W3mag':'W3', 'W4mag':'W4'}

class Query(object):
    """ RA/dec in decimal degrees, pmra, pmdec in mas
    """
    def __init__(self, ra, dec, pmra=0., pmdec=0., epoch=2000., radius=5*u.arcsec):
        self.ra = ra
        self.dec = dec
        self.pmra = pmra
        self.pmdec = pmdec
        self.epoch = epoch

        if type(radius) in [type(1), type(1.)]:
            self.radius = radius*u.arcsec
        else:
            self.radius = radius

        self._coords = None

    def __str__(self):
        return '({0.ra}, {0.dec}), pm=({0.pmra}, {0.pmdec}), epoch={0.epoch}, radius={0.radius}'.format(self)

    def __repr__(self):
        return ('Query(ra={0.ra}, dec={0.dec}, pmra={0.pmra}, '.format(self) + 
                'pmdec={0.pmdec}, epoch={0.epoch}, radius={0.radius})'.format(self))

    @property
    def coords(self):
        if self._coords is None:
            self._coords = SkyCoord(self.ra, self.dec, unit='deg')
        return self._coords
    
