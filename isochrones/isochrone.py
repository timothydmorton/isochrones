from .config import on_rtd

from .logger import getLogger

try:
    import holoviews as hv  # noqa
except ImportError:
    getLogger().warning("Holoviews not imported. Some visualizations will not be available.")
    pass

if not on_rtd:
    from .models import ModelGridInterpolator


def get_ichrone(models, bands=None, default=False, tracks=False, basic=False, **kwargs):
    """Gets Isochrone Object by name, or type, with the right bands

    If `default` is `True`, then will set bands
    to be the union of bands and default_bands
    """
    if not bands:
        bands = None

    if isinstance(models, ModelGridInterpolator):
        return models

    if type(models) is type(type):
        ichrone = models(bands)
    elif models == "mist":
        if tracks:
            from isochrones.mist import MIST_EvolutionTrack

            ichrone = MIST_EvolutionTrack(bands=bands, **kwargs)
        else:
            if basic:
                from isochrones.mist import MISTBasic_Isochrone

                ichrone = MISTBasic_Isochrone(bands=bands, **kwargs)
            else:
                from isochrones.mist import MIST_Isochrone

                ichrone = MIST_Isochrone(bands=bands, **kwargs)
    else:
        raise ValueError("Unknown stellar models: {}".format(models))
    return ichrone
