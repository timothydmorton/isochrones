from .version import __version__

try:
    __ISOCHRONES_SETUP__
except NameError:
    __ISOCHRONES_SETUP__ = False

if not __ISOCHRONES_SETUP__:
    from .isochrone import get_ichrone
    from .starmodel import StarModel, SingleStarModel, BinaryStarModel, TripleStarModel
