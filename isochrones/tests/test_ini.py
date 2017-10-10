import os
import numpy as np

from isochrones.dartmouth import Dartmouth_Isochrone
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel, BinaryStarModel, TripleStarModel

FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))

DAR = Dartmouth_Isochrone()
MIST = MIST_Isochrone()

def test_ini():
    _check_ini(DAR)
    _check_ini(MIST)


#################

def _check_ini(ic):
    single_dirs = ['star1']
    binary_dirs = ['star2']
    triple_dirs = ['star3', 'star4']

    for d in single_dirs:
        SingleCheck().check(ic, os.path.join(FOLDER, d))
    for d in binary_dirs:
        BinaryCheck().check(ic, os.path.join(FOLDER, d))
        BinaryCheck_Unassoc().check(ic, os.path.join(FOLDER, d))
    for d in triple_dirs:
        TripleCheck().check(ic, os.path.join(FOLDER, d))
        TripleCheck_Unassoc1().check(ic, os.path.join(FOLDER, d))
        TripleCheck_Unassoc2().check(ic, os.path.join(FOLDER, d))

    # _ini1(ic)
    # _ini2(ic)
    # _ini3(ic)
    # _ini3_2(ic)
    # _ini4(ic)

class IniCheck(object):
    index = 0

    def get_mod(self, ic, folder):
        return StarModel.from_ini(ic, folder=folder, index=self.index)

    def check_asserts(self, mod):
        assert self.n_params == len(self.pars)
        assert mod.n_params == self.n_params
        assert mod.obs.systems == self.systems
        assert mod.obs.Nstars == self.Nstars
        assert np.isfinite(mod.lnlike(self.pars))

    def check_p0(self, mod):
        p0 = mod.emcee_p0(200)
        nbad = 0
        for i,p in enumerate(p0):
            if not np.isfinite(mod.lnpost(p)):
                print(p)
                nbad += 1
        assert nbad==0        

    def check(self, ic, folder):
        mod = self.get_mod(ic, folder)
        self.check_asserts(mod)
        self.check_p0(mod)

        if hasattr(self, 'get_mod_special'):
            mod = self.get_mod_special(ic, folder)
            self.check_asserts(mod)

class SingleCheck(IniCheck):
    pars = [1.0, 9.4, 0.0, 100, 0.2]
    n_params = 5
    systems = [0]
    Nstars = {0:1}

class BinaryCheck(IniCheck):
    pars = [1.0, 0.5, 9.4, 0.0, 100, 0.2]
    n_params = 6
    systems = [0]
    Nstars = {0:2}

    def get_mod_special(self, ic, folder):
        return BinaryStarModel(ic, folder=folder)

class BinaryCheck_Unassoc(IniCheck):
    pars = [1.0, 9.4, 0.0, 100, 0.2, 
            0.8, 9.7, 0.1, 300, 0.3]
    n_params = 10
    index = [0, 1]
    systems = [0, 1]
    Nstars = {0:1, 1:1}

class TripleCheck(IniCheck):
    pars = [1.0, 0.8, 0.5, 9.4, 0.0, 100, 0.2]
    n_params = 7
    systems = [0]
    Nstars = {0:3}

    def get_mod_special(self, ic, folder):
        return TripleStarModel(ic, folder=folder)


class TripleCheck_Unassoc1(IniCheck):
    pars = [1.0, 0.8, 9.4, 0.0, 100, 0.2, 
            1.0, 9.7, 0.0, 200, 0.5]
    n_params = 11
    index = [0, 0, 1]
    systems = [0, 1]
    Nstars = {0:2, 1:1}

class TripleCheck_Unassoc2(IniCheck):
    pars = [1.0, 9.4, 0.0, 100, 0.2, 
            1.0, 0.8, 9.7, 0.0, 200, 0.5]
    n_params = 11
    index = [0, 1, 1]
    systems = [0, 1]
    Nstars = {0:1, 1:2}


