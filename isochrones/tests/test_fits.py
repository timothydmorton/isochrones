import os, glob
import numpy as np
import tempfile

from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel

mnest = True
try:
    import pymultinest
except:
    logging.warning('No PyMultiNest; fits will use emcee')
    mnest = False

chainsdir = tempfile.gettempdir()

props = dict(Teff=(5800, 100), logg=(4.5, 0.1), 
             B=(5.7,0.05), V=(5.0, 0.05))

def test_fitting():
    _check_fitting(StarModel(Dartmouth_Isochrone, **props))
    _check_fitting(StarModel(MIST_Isochrone, **props))


###############

def _check_fitting(mod):
    if mnest:
        _fit_mnest(mod)
    _fit_emcee(mod)

def _fit_mnest(mod):
    basename = '{}/{}-'.format(chainsdir,np.random.randint(1000000))
    mod.fit_multinest(n_live_points=5, max_iter=50,basename=basename,
            verbose=False)
    foo = mod.mnest_analyzer
    files = glob.glob('{}*'.format(basename))
    for f in files:
        os.remove(f)
    
def _fit_emcee(mod):
    mod.use_emcee = True
    mod.fit_mcmc(nburn=20, niter=10, ninitial=10)
    mod.samples
