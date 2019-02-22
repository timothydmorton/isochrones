from .grid import ParsecModelGrid
from .isochrone import Parsec_Isochrone

def download_grids():
    return ParsecModelGrid.download_grids()