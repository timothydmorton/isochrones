from .grid import MISTModelGrid
from .isochrone import MIST_Isochrone

def download_grids():
    return MISTModelGrid.download_grids()