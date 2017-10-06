from .isochrone import Dartmouth_Isochrone, Dartmouth_FastIsochrone
from .grid import DartmouthModelGrid

def download_grids():
    return DartmouthModelGrid.download_grids()