
from ..isochrone import AltIsochrone
from .grid import MISTModelGrid

class MIST_Isochrone(AltIsochrone):
    name = 'mist'
    age_col = 1
    feh_col = 7
    mass_col = 2
    loggTeff_col = 3
    logg_col = 4
    logL_col = 5
    modelgrid = MISTModelGrid
    default_bands = ['G','B','V','J','H','K','W1','W2','W3','g','r','i','z','Kepler']

    def Z_surf(self, mass, age, feh):
        return self.interp_value(mass, age, feh, 6)


