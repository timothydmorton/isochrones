from isochrones.extinction.grid import get_extinction_grid

def test_extinction_grid():
    bands = ['G', 'B', 'J', 'g', 'W1']
    A_grids = {b : get_extinction_grid(b, overwrite=True) for b in bands}
