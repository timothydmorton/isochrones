import pandas as pd
import os
import logging
import tarfile

from .utils import download_file
from .interp import DFInterpolator


class Grid(object):
    is_full = False

    def __init__(self, **kwargs):

        if hasattr(self, 'default_kwargs'):
            self.kwargs = self.default_kwargs.copy()
        else:
            self.kwargs = {}
        self.kwargs.update(kwargs)

        self._df = None
        self._interp = None

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

    def read_hdf(self):
        h5file = self.hdf_filename
        try:
            df = pd.read_hdf(h5file, 'df')
        except FileNotFoundError:
            df = self.write_hdf()
        return df

    def write_hdf(self):
        df = self.get_df()
        h5file = self.hdf_filename
        df.to_hdf(h5file, 'df')
        logging.info('{} written.'.format(h5file))
        return df

    @property
    def df(self):
        if self._df is None:
            self._df = self.get_df()
        return self._df

    @property
    def interp(self):
        if self._interp is None:
            self._interp = DFInterpolator(self.df, is_full=self.is_full)
        return self._interp
