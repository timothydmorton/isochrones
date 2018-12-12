
from ..isochrone import FastIsochrone
from .grid import MISTModelGrid

class MIST_Isochrone(FastIsochrone):
    """MESA Isochrones and Stellar Tracks

    :param bands: (optional)
        List of desired photometric bands.  Default list of bands is
        ``['G','B','V','J','H','K','W1','W2','W3','g','r','i','z','Kepler']``.
        Here ``B`` and ``V`` are Tycho-2 mags, `griz` are SDSS, and ``G`` is
        Gaia G-band.

    Details of models are `here <http://waps.cfa.harvard.edu/MIST/>`_.

    """
    name = 'mist'
    age_col = 1
    feh_col = 7
    mass_col = 2
    loggTeff_col = 3
    logg_col = 4
    logL_col = 5
    modelgrid = MISTModelGrid
    default_bands = ('G','B','V','J','H','K','W1','W2','W3','g','r','i','z','Kepler', 'TESS', 'BP', 'RP')

    def __init__(self, *args, **kwargs):
        self.version = kwargs.get('version', MISTModelGrid.default_kwargs['version'])
        if self.version >= '1.1':
            self.mass_col = 3
            self.loggTeff_col = 4
            self.logg_col = 5
            self.logL_col = 6
            self.feh_col = 7

        super().__init__(*args, **kwargs)

    def Z_surf(self, mass, age, feh):
        return self.interp_value(mass, age, feh, 6)
