
import numpy as np
from isochrones.dartmouth import Dartmouth_Isochrone
import tempfile
import os, shutil, glob
import logging

mnest = True
try:
    import pymultinest
except:
    logging.warning('No PyMultiNest; fits will use emcee')
    mnest = False

DAR = Dartmouth_Isochrone()

props = dict(Teff=(5800, 100), logg=(4.5, 0.1), 
             B=(5.7,0.05), V=(5.0, 0.05))

chainsdir = tempfile.gettempdir()

def test_dartmouth():
    DAR.radius(1., 9.5, -0.2)
    DAR.radius(np.ones(100), 9.5, 0.0)
    DAR.mag['J'](1, 9.5, 0.1, 500, 0.2)

def fit_mnest(mod):
    basename = '{}/{}-'.format(chainsdir,np.random.randint(1000000))
    mod.fit_multinest(n_live_points=5, max_iter=20,basename=basename,
            verbose=False)
    foo = mod.mnest_analyzer
    files = glob.glob('{}*'.format(basename))
    for f in files:
        os.remove(f)
    
def fit_emcee(mod):
    mod.use_emcee = True
    mod.fit_mcmc(nburn=20, niter=10, ninitial=10)
    mod.samples

def test_single():
    from isochrones import StarModel
    mod = StarModel(DAR, **props)
    if mnest:
        fit_mnest(mod)
    fit_emcee(mod)

"""
def test_binary():
    from isochrones import BinaryStarModel
    mod = BinaryStarModel(DAR, **props)
    if mnest:
        fit_mnest(mod)
    fit_emcee(mod)

def test_triple():
    from isochrones import TripleStarModel
    mod = TripleStarModel(DAR, **props)
    if mnest:
        fit_mnest(mod)
    fit_emcee(mod)
"""
