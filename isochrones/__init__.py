__version__ = '0.8.1'

try:
    __ISOCHRONES_SETUP__
except NameError:
    __ISOCHRONES_SETUP__ = False

if not __ISOCHRONES_SETUP__:
    __all__ = ['dartmouth','basti','padova',
               'Isochrone', 'StarModel', 'BinaryStarModel',
               'TripleStarModel']
    from .isochrone import Isochrone
    from .starmodel import StarModel, BinaryStarModel, TripleStarModel
     
