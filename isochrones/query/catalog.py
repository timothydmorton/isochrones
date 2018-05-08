import pandas as pd
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

class Catalog(object):
    """ Base class for results from catalog queries
    """
    _distance_column = '_r'

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
        df = self.df.sort_values(by=self._distance_column)
        return df.iloc[0]

    @property
    def brightest(self):
        band = list(self.bands.keys())[0]
        df = self.df.sort_values(by=band)
        return df.iloc[0]

    def get_id(self, brightest=False):
        if brightest:
            row = self.brightest
        else:
            row = self.closest

        return row[self.id_column]

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
