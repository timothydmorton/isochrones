__version__ = '0.9.0-150-g2f34c34'

try:
    __ISOCHRONES_SETUP__
except NameError:
    __ISOCHRONES_SETUP__ = False

if not __ISOCHRONES_SETUP__:
    __all__ = ['dartmouth','basti','padova',
               'Isochrone', 'StarModel']
    from .isochrone import Isochrone
    from .starmodel import StarModel #, BinaryStarModel, TripleStarModel
     
