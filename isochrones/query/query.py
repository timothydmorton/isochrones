from astropy.coordinates import SkyCoord
import astropy.units as u

class EmptyQueryError(ValueError):
    pass

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
    
