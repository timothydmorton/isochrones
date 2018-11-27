__version__ = '1.2.2dev'

try:
    __ISOCHRONES_SETUP__
except NameError:
    __ISOCHRONES_SETUP__ = False

if not __ISOCHRONES_SETUP__:
    from .isochrone import get_ichrone
    from .starmodel import StarModel
