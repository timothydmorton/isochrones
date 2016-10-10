import os,re, glob
import numpy as np
import pandas as pd
import logging

from ..config import ISOCHRONES

class ModelGrid(object):
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
    zenodo_record = 159426
    zenodo_files = ('dartmouth.tgz', 'dartmouth.tri')
    master_tarball_file = 'dartmouth.tgz'

    def __init__(self, bands, **kwargs):
        self.bands = bands
        self.kwargs = kwargs

        for k,v in self.default_kwargs.items():
            if k not in self.kwargs:
                self.kwargs[k] = v            

        self._df = None

    def get_band(self, b):
        """Defines what a "shortcut" band name refers to.  Returns phot_system, band

        """
        # Default to SDSS for these
        if b in ['u','g','r','i','z']:
            sys = 'SDSSugriz'
            band = 'sdss_{}'.format(b)
        elif b in ['U','B','V','R','I','J','H','Ks']:
            sys = 'UBVRIJHKsKp'
            band = b
        elif b=='K':
            sys = 'UBVRIJHKsKp'
            band = 'Ks'
        elif b in ['kep','Kepler','Kp']:
            sys = 'UBVRIJHKsKp'
            band = 'Kp'
        elif b in ['W1','W2','W3','W4']:
            sys = 'WISE'
            band = b
        else:
            m = re.match('([a-zA-Z]+)_([a-zA-Z_]+)',b)
            if m:
                if m.group(1) in self.phot_systems:
                    sys = m.group(1)
                    if sys=='LSST':
                        band = b
                    else:
                        band = m.group(2)
                elif m.group(1) in ['UK','UKIRT']:
                    sys = 'UKIDSS'
                    band = m.group(2)
        return sys, band

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
                grids[s] = self.get_hdf(s, **self.kwargs)
            if 'MMo' not in df:
                df[list(self.common_columns)] = grids[s][list(self.common_columns)]
            col = grids[s][b]
            n_nan = np.isnan(col).sum()
            if n_nan > 0:
                logging.debug('{} NANs in {} column'.format(n_nan, b))
            df.loc[:, bnd] = col.values #dunno why it has to be this way; something
                                        # funny with indexing.

        return df


    def download_grids(self, overwrite=True):
        record = self.zenodo_record

        paths = []
        urls = []
        for f in self.zenodo_files:
            paths.append(os.path.join(ISOCHRONES, f))
            urls.append('https://zenodo.org/record/{}/files/{}'.format(record, f))

        from six.moves import urllib
        print('Downloading {} stellar model data (should happen only once)...'.format(self.name))

        for path, url in zip(paths, urls):
            if os.path.exists(path):
                if overwrite:
                    os.remove(path)
                else:
                    continue
            urllib.request.urlretrieve(url, path)


    def extract_master_tarball(self):
        """Unpack tarball of tarballs
        """
        with tarfile.open(os.path.join(ISOCHRONES, self.master_tarball_file)) as tar:
            logging.info('Extracting {}...'.format(self.master_tarball_file))
            tar.extractall(ISOCHRONES)

    # Specific
    def phot_tarball_file(self, phot, **kwargs):
        return os.path.join(self.datadir, '{}.tgz'.format(phot))

    def extract_phot_tarball(self, phot, **kwargs):
        phot_tarball = self.phot_tarball_file(phot)
        with tarfile.open(phot_tarball) as tar:
            logging.info('Extracting {}.tgz...'.format(phot))
            tar.extractall(self.datadir)

    # Specific
    def get_filenames(self, phot, afe='afep0', y=''):
        if not os.path.exists(os.path.join(self.datadir, 'isochrones', phot)):
            if not os.path.exists(self.phot_tarball_file(phot, afe=afe, y=y)):
                extract_master_tarball()
            extract_phot_tarball(phot)

        return glob.glob('{3}/isochrones/{0}/*{1}{2}.{0}*'.format(phot,afe,y,self.datadir))

    # Specific
    def get_feh(self, filename):
        m = re.search('feh([mp])(\d+)afe', filename)
        if m:
            sign = 1 if m.group(1)=='p' else -1
            return float(m.group(2))/10 * sign
        
    # Specific
    def to_df(self, filename):
        try:
            rec = np.recfromtxt(filename,skip_header=8,names=True)
        except:
            print('Error reading {}!'.format(filename))
            raise RuntimeError
        df = pd.DataFrame(rec)
        
        n = len(df)
        ages = np.zeros(n)
        curage = 0
        i=0
        for line in open(filename):
            m = re.match('#',line)
            if m:
                m = re.match('#AGE=\s*(\d+\.\d+)\s+',line)
                if m:
                    curage=float(m.group(1))
            else:
                if re.search('\d',line):
                    ages[i]=curage
                    i+=1

        df['age'] = ages
        df['feh'] = self.get_feh(filename)
        return df

    # Specific (in sorting)
    def df_all(self, phot, afe='afep0', y=''):
        df = pd.concat([self.to_df(f) for f in self.get_filenames(phot, afe=afe, y=y)])
        return df.sort_values(by=['age','feh','MMo','EEP'])
        
    def hdf_filename(self, phot, afe='afep0', y=''):
        afe_str = '_{}'.format(afe) if afe!='afep0' else ''
        return os.path.join(self.datadir,'{}{}.h5'.format(phot, afe_str, y))

    def get_hdf(self, phot, afe='afep0', y=''):
        h5file = self.hdf_filename(phot, afe=afe, y=y)
        try:
            df = pd.read_hdf(h5file, 'df')
        except:
            df = self.write_hdf(phot, afe=afe, y=y)
        return df

    def write_hdf(self, phot, afe='afep0', y=''):
        df = self.df_all(phot, afe=afe, y=y)   
        h5file = self.hdf_filename(phot, afe=afe, y=y)
        df.to_hdf(h5file,'df')
        print('{} written.'.format(h5file))
        return df

