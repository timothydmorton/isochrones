import os
import numpy as np

from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel

FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))

DAR = Dartmouth_Isochrone()
MIST = MIST_Isochrone()

def test_ini():
    _check_ini(DAR)
    _check_ini(MIST)


#################

def _check_ini(ic):
    _ini1(ic)
    _ini2(ic)
    _ini3(ic)
    _ini3_2(ic)

def _ini1(ic):
    """ Single star
    """
    mod = StarModel.from_ini(ic, folder=os.path.join(FOLDER, 'star1'))
    assert mod.n_params == 5
    assert mod.obs.systems == [0]
    assert mod.obs.Nstars == {0:1}
    p = [1.0, 9.4, 0.0, 100, 0.2]    
    assert np.isfinite(mod.lnlike(p))

def _ini2(ic):
    """ A wide, well-resolved binary
    """
    mod = StarModel.from_ini(ic, folder=os.path.join(FOLDER, 'star2'))
    assert mod.n_params == 6
    assert mod.obs.systems == [0]
    assert mod.obs.Nstars == {0:2}
    p = [1.0, 0.5, 9.4, 0.0, 100, 0.2]    
    assert np.isfinite(mod.lnlike(p))

def _ini3(ic):
    """ A close resolved triple (unresolved in KIC, TwoMASS)

    modeled as a physically associated triple
    """
    mod = StarModel.from_ini(ic, folder=os.path.join(FOLDER, 'star3'))
    assert mod.n_params == 7
    assert mod.obs.systems == [0]
    assert mod.obs.Nstars == {0:3}
    p = [1.0, 0.8, 0.5, 9.4, 0.0, 100, 0.2]    
    assert np.isfinite(mod.lnlike(p))

def _ini3_2(ic):
    """ A close resolved triple (unresolved in KIC, TwoMASS)

    modeled as a physically associated binary plus non-associated single
    """
    mod = StarModel.from_ini(ic, folder=os.path.join(FOLDER, 'star3'),
                            index=[0,0,1])
    assert mod.n_params == 11
    assert mod.obs.systems == [0, 1]
    assert mod.obs.Nstars == {0:2, 1:1}
    p = [1.0, 0.8, 9.4, 0.0, 100, 0.2, 1.0, 9.7, 0.0, 200, 0.5]    
    assert np.isfinite(mod.lnlike(p))