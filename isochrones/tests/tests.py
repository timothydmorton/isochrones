
import tempfile
import os, shutil, glob
import logging

import numpy as np
from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones import StarModel

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

FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def test_dartmouth_basic(bands=['z', 'B', 'W3', 'LSST_r', 'J', 'UK_J']):
    dar = Dartmouth_Isochrone(bands)
    dar.radius(1., 9.5, -0.2)
    dar.radius(np.ones(100), 9.5, 0.0)
    args = (1, 9.5, 0.1, 500, 0.2)
    for b in bands:
        dar.mag[b](*args)

def test_spec():
    mod = StarModel(DAR, Teff=(5700,100), logg=(4.5, 0.1), feh=(0.0, 0.2))
    assert np.isfinite(mod.lnlike([1.0, 9.6, 0.1, 200, 0.2]))

# def test_afe():
#     dar = Dartmouth_Isochrone(afe='afem2')
#     dar.radius(1, 9.4, 0.0)

def test_fitting():
    from isochrones import StarModel
    mod = StarModel(DAR, **props)
    if mnest:
        _fit_mnest(mod)
    _fit_emcee(mod)

def test_AV():
    from isochrones.extinction import get_AV_infinity
    AV = get_AV_infinity(299.268036, 45.227428)
    assert np.isclose(AV, 1.216)

def test_ini1():
    """ Single star
    """
    mod = StarModel.from_ini(Dartmouth_Isochrone, folder=os.path.join(FOLDER, 'star1'))
    assert mod.n_params == 5
    assert mod.obs.systems == [0]
    assert mod.obs.Nstars == {0:1}
    p = [1.0, 9.4, 0.0, 100, 0.2]    
    assert np.isfinite(mod.lnlike(p))

def test_ini2():
    """ A wide, well-resolved binary
    """
    mod = StarModel.from_ini(Dartmouth_Isochrone, folder=os.path.join(FOLDER, 'star2'))
    assert mod.n_params == 6
    assert mod.obs.systems == [0]
    assert mod.obs.Nstars == {0:2}
    p = [1.0, 0.5, 9.4, 0.0, 100, 0.2]    
    assert np.isfinite(mod.lnlike(p))

def test_ini3():
    """ A close resolved triple (unresolved in KIC, TwoMASS)

    modeled as a physically associated triple
    """
    mod = StarModel.from_ini(Dartmouth_Isochrone, folder=os.path.join(FOLDER, 'star3'))
    assert mod.n_params == 7
    assert mod.obs.systems == [0]
    assert mod.obs.Nstars == {0:3}
    p = [1.0, 0.8, 0.5, 9.4, 0.0, 100, 0.2]    
    assert np.isfinite(mod.lnlike(p))

def test_ini3_2():
    """ A close resolved triple (unresolved in KIC, TwoMASS)

    modeled as a physically associated triple
    """
    mod = StarModel.from_ini(Dartmouth_Isochrone, folder=os.path.join(FOLDER, 'star3'),
                            index=[0,0,1])
    assert mod.n_params == 11
    assert mod.obs.systems == [0, 1]
    assert mod.obs.Nstars == {0:2, 1:1}
    p = [1.0, 0.8, 9.4, 0.0, 100, 0.2, 1.0, 9.7, 0.0, 200, 0.5]    
    assert np.isfinite(mod.lnlike(p))


############
def _fit_mnest(mod):
    basename = '{}/{}-'.format(chainsdir,np.random.randint(1000000))
    mod.fit_multinest(n_live_points=10, max_iter=200,basename=basename,
            verbose=False)
    foo = mod.mnest_analyzer
    files = glob.glob('{}*'.format(basename))
    for f in files:
        os.remove(f)
    
def _fit_emcee(mod):
    mod.use_emcee = True
    mod.fit_mcmc(nburn=20, niter=10, ninitial=10)
    mod.samples


