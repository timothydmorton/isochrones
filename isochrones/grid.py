import pandas as pd
import os
import logging
import tarfile

from .utils import download_file
from .interp import DFInterpolator


class Grid(object):
    is_full = False
    bounds = tuple()

    def __init__(self, **kwargs):

        if hasattr(self, 'default_kwargs'):
            self.kwargs = self.default_kwargs.copy()
        else:
            self.kwargs = {}
        self.kwargs.update(kwargs)

        self._df = None
        self._df_orig = None
        self._interp = None
        self._interp_orig = None
        self._limits = dict(self.bounds)

    def get_limits(self, prop):
        if prop not in self._limits:
            self._limits[prop] = self.df[prop].min(), self.df[prop].max()
        return self._limits[prop]

    @property
    def datadir(self):
        raise NotImplementedError

    def get_hdf_filename(self, **kwargs):
        raise NotImplementedError

    @property
    def hdf_filename(self):
        return self.get_hdf_filename()

    def get_tarball_url(self, **kwargs):
        raise NotImplementedError

    def get_tarball_file(self, **kwargs):
        raise NotImplementedError

    def download_tarball(self, **kwargs):
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        tarball = self.get_tarball_file(**kwargs)
        if not os.path.exists(tarball):
            url = self.get_tarball_url(**kwargs)
            logging.info('Downloading {}...'.format(url))
            download_file(url, tarball)

    def extract_tarball(self, **kwargs):
        tarball = self.get_tarball_file(**kwargs)
        if not os.path.exists(tarball):
            self.download_tarball(**kwargs)

        with tarfile.open(tarball) as tar:
            logging.info('Extracting {}...'.format(tarball))
            tar.extractall(self.datadir)

    def read_hdf(self, orig=False):
        h5file = self.hdf_filename
        try:
            path = 'orig' if orig else 'df'
            df = pd.read_hdf(h5file, path)
        except (FileNotFoundError, KeyError):
            df = self.write_hdf(orig=orig)
        return df

    def write_hdf(self, orig=False):
        df = self.get_df(orig=orig)
        h5file = self.hdf_filename
        path = 'orig' if orig else 'df'
        df.to_hdf(h5file, path)
        logging.info('{} written to {}.'.format(path, h5file))
        return df

    @property
    def df(self):
        if self._df is None:
            self._df = self.get_df()
        return self._df

    @property
    def df_orig(self):
        if self._df_orig is None:
            self._df_orig = self.read_hdf(orig=True)
        return self._df_orig

    @property
    def interp(self):
        if self._interp is None:
            filename = getattr(self, 'interp_grid_npz_filename', None)
            self._interp = DFInterpolator(self.df, filename=filename, is_full=self.is_full)
        return self._interp

    @property
    def interp_orig(self):
        if self._interp_orig is None:
            filename = getattr(self, 'interp_grid_orig_npz_filename', None)
            self._interp_orig = DFInterpolator(self.df_orig, filename=filename, is_full=self.is_full)
        return self._interp_orig

