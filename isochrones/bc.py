import pandas as pd
import os
import glob

from .config import ISOCHRONES
from .grid import Grid


class BolometricCorrectionGrid(Grid):
    """Bolometric corrections in different bands, as a function of stuff

    Stores bolometric corrections computed on a grid of stellar atmospheric
    parameters (Teff, logg, [Fe/H]), Av, and Rv.

    Specific implementations of this grid should subclass this
    (e.g., `MISTBolometricCorrectionGrid`).

    Parameters
    ----------
    bands : list(str)
        List of band names, each parsed with `get_band` method.
        Tables are downloaded when requested.
    """

    index_cols = ('Teff', 'logg', '[Fe/H]', 'Av', 'Rv')
    name = None
    is_full = True

    def __init__(self, bands=None):

        self.bands = bands if bands is not None else list(self.default_bands)

        self._band_map = None
        self._phot_systems = None

        self._df = None
        self._interp = None

    def get_band(self, *args, **kwargs):
        return NotImplementedError

    def _make_band_map(self):
        phot_systems = set()
        band_map = {}
        for b in self.bands:
            phot, band = self.get_band(b)
            phot_systems.add(phot)
            band_map[b] = band
        self._band_map = band_map
        self._phot_systems = phot_systems

    @property
    def band_map(self):
        if self._band_map is None:
            self._make_band_map()
        return self._band_map

    @property
    def phot_systems(self):
        if self._phot_systems is None:
            self._make_band_map()
        return self._phot_systems

    @property
    def datadir(self):
        return os.path.join(ISOCHRONES, 'BC', self.name)

    def get_filename(self, phot, feh):
        rootdir = self.datadir
        sign_str = 'm' if feh < 0 else 'p'
        filename = 'feh{0}{1:03.0f}.{2}'.format(sign_str, abs(feh)*100, phot)
        return os.path.join(rootdir, filename)

    def parse_table(self, filename):
        """Reads text table into dataframe
        """
        with open(filename) as fin:
            for i, line in enumerate(fin):
                if i == 5:
                    names = line[1:].split()
                    break
        return pd.read_csv(filename, names=names, delim_whitespace=True, comment='#',
                           index_col=self.index_cols)

    def get_table(self, phot, feh):
        return self.parse_table(self.get_filename(phot, feh))

    def get_hdf_filename(self, phot):
        return os.path.join(self.datadir, '{}.h5'.format(phot))

    def get_tarball_url(self, phot):
        url = 'http://waps.cfa.harvard.edu/MIST/BC_tables/{}.txz'.format(phot)
        return url

    def get_tarball_file(self, phot):
        return os.path.join(self.datadir, '{}.txz'.format(phot))

    def get_df(self):
        df_all = pd.DataFrame()
        for phot in self.phot_systems:
            hdf_filename = self.get_hdf_filename(phot=phot)
            if not os.path.exists(hdf_filename):
                filenames = glob.glob(os.path.join(self.datadir, '*.{}'.format(phot)))
                if not filenames:
                    self.extract_tarball(phot=phot)
                    filenames = glob.glob(os.path.join(self.datadir, '*.{}'.format(phot)))
                df = pd.concat([self.parse_table(f) for f in filenames]).sort_index()
                df.to_hdf(hdf_filename, 'df')
            df = pd.read_hdf(hdf_filename)
            df_all = pd.concat([df_all, df], axis=1)

        df_all = df_all.rename(columns={v: k for k, v in self.band_map.items()})
        for col in df_all.columns:
            if col not in self.bands:
                del df_all[col]

        return df_all
