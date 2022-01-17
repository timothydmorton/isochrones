import os
import glob
import numpy as np
import tempfile
import tables as tb

from pandas.testing import assert_frame_equal
from flaky import flaky

from isochrones.mist import MIST_Isochrone
from isochrones import StarModel
from isochrones.starfit import starfit
from isochrones.logger import getLogger

mnest = True
try:
    import pymultinest  # noqa
except ImportError:
    getLogger().warning("No PyMultiNest; fits will use emcee")
    mnest = False

chainsdir = tempfile.gettempdir()

props = dict(Teff=(5800, 100), logg=(4.5, 0.1), J=(3.58, 0.05), K=(3.22, 0.05))


def test_fitting():
    mod_mist = _check_fitting(StarModel(MIST_Isochrone, **props))

    _check_saving(mod_mist)


@flaky
def test_starfit():
    rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    testdir = os.path.join(rootdir, "star1")
    if mnest:
        basename = "{}/{}-".format(chainsdir, np.random.randint(1000000))
        kwargs = dict(n_live_points=20, max_iter=100, basename=basename, verbose=False)
        getLogger().info("Testing starfit function with multinest...")
    else:
        kwargs = dict(nburn=20, niter=20, ninitial=10)
        getLogger().info("Testing starfit function with emcee...")

    mod, _ = starfit(testdir, overwrite=True, use_emcee=not mnest, no_plots=True, **kwargs)

    mod.samples

    if mnest:
        files = glob.glob("{}*".format(basename))
        for f in files:
            os.remove(f)


###############


def _check_saving(mod):
    filename = os.path.join(chainsdir, "{}.h5".format(np.random.randint(1000000)))
    mod.save_hdf(filename)
    assert len(tb.file._open_files.get_handlers_by_name(filename)) == 0

    newmod = StarModel.load_hdf(filename)
    assert len(tb.file._open_files.get_handlers_by_name(filename)) == 0

    assert_frame_equal(mod.samples, newmod.samples)
    assert mod.ic.bands == newmod.ic.bands

    os.remove(filename)


def _check_fitting(mod):
    _fit_emcee(mod)
    if mnest:
        _fit_mnest(mod)
    return mod


def _fit_mnest(mod):
    basename = "{}/{}-".format(chainsdir, np.random.randint(1000000))
    mod.fit_multinest(n_live_points=5, max_iter=50, basename=basename, verbose=False)
    mod.mnest_analyzer
    files = glob.glob("{}*".format(basename))
    for f in files:
        os.remove(f)


def _fit_emcee(mod):
    mod.use_emcee = True
    mod.fit_mcmc(nburn=20, niter=20, ninitial=20)
    mod.samples
