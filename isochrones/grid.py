import os,re, glob
import tarfile
import logging

from .config import ISOCHRONES, on_rtd
from .utils import download_file

if not on_rtd:
    import numpy as np
    import pandas as pd


class ModelGrid(object):
    """Base class for Model Grids.

    Subclasses must implement the following (shown below is the Dartmouth example)::

        name = 'dartmouth'
        common_columns = ('EEP', 'MMo', 'LogTeff', 'LogG', 'LogLLo', 'age', 'feh')
        phot_systems = ('SDSSugriz','UBVRIJHKsKp','WISE','LSST','UKIDSS')
        phot_bands = dict(SDSSugriz=['sdss_z', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_g'],
                      UBVRIJHKsKp=['B', 'I', 'H', 'J', 'Ks', 'R', 'U', 'V', 'D51', 'Kp'],
                      WISE=['W4', 'W3', 'W2', 'W1'],
                      LSST=['LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'LSST_g', 'LSST_i'],
                      UKIDSS=['Y', 'H', 'K', 'J', 'Z'])

        default_kwargs = {'afe':'afep0', 'y':''}
        datadir = os.path.join(ISOCHRONES, 'dartmouth')
        zenodo_record = 159426  # if you want to store data here
        zenodo_files = ('dartmouth.tgz', 'dartmouth.tri') # again, if desired
        master_tarball_file = 'dartmouth.tgz'

    Subclasses also must implement the following methods:

    `get_band`, `phot_tarball_file`, `get_filenames`, `get_feh`,
    `to_df`, `hdf_filename`.  See :class:`DartmouthModelGrid`
    and :class:`MISTModelGrid` for details.
    """

    def __init__(self, bands=None, **kwargs):
        if bands is None:
            bands = self.default_bands

        self.bands = sorted(bands)
        self.kwargs = self.default_kwargs.copy()
        self.kwargs.update(kwargs)

        self._df = None

    @classmethod
    def get_common_columns(self, **kwargs):
        return self.common_columns

    @classmethod
    def get_band(cls, b):
        """Must defines what a "shortcut" band name refers to.

        :param: b (string)
            Band name.

        :return: phot_system, band
            ``b`` maps to the band defined by ``phot_system`` as ``band``.
        """

        raise NotImplementedError

    def phot_tarball_file(self, phot, **kwargs):
        """Returns name of tarball file for given phot system and kwargs
        """
        raise NotImplementedError

    def get_filenames(self, phot, **kwargs):
        """ Returns list of all filenames corresponding to phot system and kwargs.
        """
        raise NotImplementedError

    @classmethod
    def get_feh(cls, filename):
        """Parse [Fe/H] from filename (returns float)
        """
        raise NotImplementedError

    @classmethod
    def to_df(cls, filename):
        """Parses specific file to a pandas DataFrame
        """
        raise NotImplementedError

    def hdf_filename(cls, phot):
        """Returns HDF filename of parsed/stored phot system
        """
        raise NotImplementedError

    @property
    def df(self):
        if self._df is None:
            self._df = self._get_df()

        return self._df

    def _get_df(self):
        """Returns stellar model grid with desired bandpasses and with standard column names

        bands must be iterable, and are parsed according to :func:``get_band``
        """
        grids = {}
        df = pd.DataFrame()
        for bnd in self.bands:
            s,b = self.get_band(bnd, **self.kwargs)
            logging.debug('loading {} band from {}'.format(b,s))
            if s not in grids:
                grids[s] = self.get_hdf(s)
            if self.common_columns[0] not in df:
                df[list(self.common_columns)] = grids[s][list(self.common_columns)]
            col = grids[s][b]
            n_nan = np.isnan(col).sum()
            if n_nan > 0:
                logging.debug('{} NANs in {} column'.format(n_nan, b))
            df.loc[:, bnd] = col.values #dunno why it has to be this way; something
                                        # funny with indexing.

        return df

    @classmethod
    def download_grids(cls, overwrite=False):
        record = cls.zenodo_record
        paths = []
        urls = []
        for f in cls.zenodo_files:
            paths.append(os.path.join(ISOCHRONES, f))
            urls.append('https://zenodo.org/record/{}/files/{}'.format(record, f))

        logging.info('Downloading files for {} model grid: {}...'.format(cls.name, cls.zenodo_files))
        for path, url in zip(paths, urls):
            if os.path.exists(path):
                if overwrite:
                    os.remove(path)
                else:
                    logging.info('{} exists; not downloading.'.format(path))
                    continue
            download_file(url, path)
        cls.verify_grids()

    @classmethod
    def verify_grids(cls):
        import hashlib
        files = [os.path.join(ISOCHRONES, f) for f in cls.zenodo_files]
        good = True
        for f, md5 in zip(files, cls.zenodo_md5):
            if not os.path.exists(f):
                cls.download_grids()
                # raise RuntimeError('{0} does not exist.  Run "import isochrones.{1}; isochrones.{1}.download_grids()" to download.'.format(f, cls.name))
            if hashlib.md5(open(f,'rb').read()).hexdigest() != md5:
                raise RuntimeError('{0} is wrong/corrupted.  Delete {0} and try again.'.format(f))
                good = False
            else:
                logging.debug('{} verified.'.format(f))
        return good


    @classmethod
    def extract_master_tarball(cls):
        """Unpack tarball of tarballs
        """
        if not os.path.exists(cls.master_tarball_file):
            cls.download_grids()

        with tarfile.open(os.path.join(ISOCHRONES, cls.master_tarball_file)) as tar:
            logging.info('Extracting {}...'.format(cls.master_tarball_file))
            tar.extractall(ISOCHRONES)

    def phot_tarball_url(self, phot):
        url = '{}/{}.tgz'.format(self.extra_url_base, phot)
        return url

    def extract_phot_tarball(self, phot, **kwargs):
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        phot_tarball = self.phot_tarball_file(phot)
        if not os.path.exists(phot_tarball):
            url = self.phot_tarball_url(phot)
            logging.info('Downloading {}...'.format(url))
            download_file(url, phot_tarball)
        with tarfile.open(phot_tarball) as tar:
            logging.info('Extracting {}.tgz...'.format(phot))
            tar.extractall(self.datadir)

    def df_all(self, phot):
        """Subclasses may want to sort this
        """
        df = pd.concat([self.to_df(f) for f in self.get_filenames(phot)])
        return df

    def get_hdf(self, phot):
        h5file = self.hdf_filename(phot)
        try:
            df = pd.read_hdf(h5file, 'df')
        except:
            df = self.write_hdf(phot)
        return df

    def write_hdf(self, phot):
        df = self.df_all(phot)
        h5file = self.hdf_filename(phot)
        df.to_hdf(h5file,'df')
        logging.info('{} written.'.format(h5file))
        return df
