import os,re, glob
import tarfile
import numpy as np
import pandas as pd
import logging

from .config import ISOCHRONES

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
    def __init__(self, bands, **kwargs):
        self.bands = sorted(bands)
        self.kwargs = kwargs

        for k,v in self.default_kwargs.items():
            if k not in self.kwargs:
                self.kwargs[k] = v            

        self._df = None

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
            s,b = self.get_band(bnd)
            logging.debug('loading {} band from {}'.format(b,s))
            if s not in grids:
                grids[s] = self.get_hdf(s)
            if 'MMo' not in df:
                df[list(self.common_columns)] = grids[s][list(self.common_columns)]
            col = grids[s][b]
            n_nan = np.isnan(col).sum()
            if n_nan > 0:
                logging.debug('{} NANs in {} column'.format(n_nan, b))
            df.loc[:, bnd] = col.values #dunno why it has to be this way; something
                                        # funny with indexing.

        return df

    @classmethod
    def download_grids(cls, overwrite=True):
        record = cls.zenodo_record

        paths = []
        urls = []
        for f in cls.zenodo_files:
            paths.append(os.path.join(ISOCHRONES, f))
            urls.append('https://zenodo.org/record/{}/files/{}'.format(record, f))

        from six.moves import urllib
        print('Downloading {} stellar model data (should happen only once)...'.format(cls.name))

        for path, url in zip(paths, urls):
            if os.path.exists(path):
                if overwrite:
                    os.remove(path)
                else:
                    continue
            urllib.request.urlretrieve(url, path)

    @classmethod
    def extract_master_tarball(cls):
        """Unpack tarball of tarballs
        """
        with tarfile.open(os.path.join(ISOCHRONES, cls.master_tarball_file)) as tar:
            logging.info('Extracting {}...'.format(cls.master_tarball_file))
            tar.extractall(ISOCHRONES)

    @classmethod
    def extract_phot_tarball(cls, phot, **kwargs):
        phot_tarball = cls.phot_tarball_file(phot)
        with tarfile.open(phot_tarball) as tar:
            logging.info('Extracting {}.tgz...'.format(phot))
            tar.extractall(cls.datadir)

    def df_all(self, phot):
        """Subclasses may want to sort this
        """
        df = pd.concat([self.to_df(f) for f in self.get_filenames(phot, **self.kwargs)])
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
        print('{} written.'.format(h5file))
        return df

