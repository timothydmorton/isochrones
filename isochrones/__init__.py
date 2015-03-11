__version__ = '0.5-beta'

try:
    __ISOCHRONES_SETUP__
except NameError:
    __ISOCHRONES_SETUP__ = False

if not __ISOCHRONES_SETUP__:
    __all__ = ['dartmouth','basti','padova']
    from .isochrone import Isochrone
    from .starmodel import StarModel
     
