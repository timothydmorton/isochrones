from isochrones.extinction.grid import get_extinction_grid

def test_extinction_grid():
    bands = ['G', 'J', 'W1']
    A_grids = {b : get_extinction_grid(b, overwrite=True) for b in bands}
    A_grids2 = {b : get_extinction_grid(b, extinction='fitz', overwrite=True) 
                for b in bands}
