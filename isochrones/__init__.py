__version__ = '1.2.2'

try:
    __ISOCHRONES_SETUP__
except NameError:
    __ISOCHRONES_SETUP__ = False

if not __ISOCHRONES_SETUP__:
    __all__ = ['get_ichrone', 'Isochrone', 'StarModel', 'ModelGrid']
    from .isochrone import Isochrone, get_ichrone
    from .starmodel import StarModel, BinaryStarModel, TripleStarModel
    from .grid import ModelGrid
