
from ..isochrone import FastIsochrone
from .grid import ParsecModelGrid
#Zini  Age      Mini            Mass   logL    logTe  logg  label   McoreTP C_O  period0 period1 pmode  Mloss  tau1m   X   Y   Xc  Xn  Xo  Cexcess  Z 	 mbolmag  umag    gmag    rmag    imag    zmag

class Parsec_Isochrone(FastIsochrone):
    """Parsec Tracks

    :param bands: (optional)
        List of desired photometric bands.  Default list of bands is
        ``['G','B','V','J','H','K','W1','W2','W3','g','r','i','z','Kepler']``.
        Here ``B`` and ``V`` are Tycho-2 mags, `griz` are SDSS, and ``G`` is
        Gaia G-band.


    """
    name = 'parsec'
    age_col = 1
    feh_col = 0
    mass_col = 2
    loggTeff_col = 5
    logg_col = 6
    logL_col = 4
    modelgrid = ParsecModelGrid
    default_bands = ('G','BP','RP','J','H','K','W1','W2','W3','g','r','i','z')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Z_surf(self, mass, age, feh):
        return self.interp_value(mass, age, feh, 6)

