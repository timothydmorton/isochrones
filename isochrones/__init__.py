__version__ = '1.0'

try:
    __ISOCHRONES_SETUP__
except NameError:
    __ISOCHRONES_SETUP__ = False

if not __ISOCHRONES_SETUP__:
    __all__ = ['get_ichrone', 'Isochrone', 'StarModel']
    from .isochrone import Isochrone, get_ichrone
    from .starmodel import StarModel 
