import os,re, glob
import numpy as np
import pandas as pd
import logging
import tarfile
from distutils.version import StrictVersion

from ..config import ISOCHRONES
from ..grid import ModelGrid

class ParsecModelGrid(ModelGrid):
    name = 'parsec'
    common_columns = ('Zini', 'Age', 'Mini', 'Mass','logL', 'logTe', 'logg')

    phot_systems = ('opt', 'gaia', 'ir', 'sdss')

    phot_bands = dict(opt=['Umag', 'Bmag', 'Vmag',
                        'Rmag', 'Imag','Jmag', 'Hmag', 'Kmag'],
                      gaia=['Gmag', 'G_BPmag', 'G_RPmag'],
                      ir=['IRAC_3.6mag ', 'IRAC_4.5mag', 'IRAC_5.8mag', 'IRAC_8.0mag', 'MIPS_24mag', 'W1mag', 'W2mag', 'W3mag', 'W4mag'],
                      sdss=['umag', 'gmag', 'rmag', 'imag', 'zmag'])

    default_kwargs = {'version':'1.0'}
    datadir = os.path.join(ISOCHRONES, 'parsec')
    #zenodo_record = 161241
    #zenodo_files = ()#('mist.tgz',)
    #zenodo_md5 = ('0deaaca2836c7148c27ce5ba5bbdfe59',)
    #master_tarball_file = 'parsec.tgz'

    default_bands = ('G','BP','RP','J','H','K','W1','W2','W3','g','r','i','z')

    def __init__(self, *args, **kwargs):
        version = kwargs.get('version', self.default_kwargs['version'])
        version = StrictVersion(str(version))

        super().__init__(*args, **kwargs)

    @classmethod
    def get_common_columns(cls, version=None, **kwargs):
        if version is None:
            version = cls.default_kwargs['version']

        version = StrictVersion(str(version))
        return ('Zini', 'Age', 'Mini', 'Mass','logL', 'logTe', 'logg')


    @property
    def version(self):
        return StrictVersion(str(self.kwargs['version']))

    @property
    def common_columns(self):
        return self.get_common_columns(self.version)
        
    def phot_tarball_url(self, phot):
        if phot=='ir':   url = 'https://www.dropbox.com/s/rlb5ifn2htbgn5l/ir.tar.gz?dl=1'
        if phot=='sdss': url = 'https://www.dropbox.com/s/6ep3g9ey8j6waxl/sdss.tar.gz?dl=1'
        if phot=='gaia': url = 'https://www.dropbox.com/s/120hxb4n88apaov/gaia.tar.gz?dl=1'
        if phot=='opt':  url = 'https://www.dropbox.com/s/vdu58x4pfjbuhsz/opt.tar.gz?dl=1'
        return url

    @classmethod
    def get_band(cls, b, **kwargs):
        """Defines what a "shortcut" band name refers to.  Returns phot_system, band

        """
        phot = None

        # Default to SDSS for these
        if b in ['u','g','r','i','z']:
            phot = 'sdss'
            band = '{}mag'.format(b)
        elif b in ['U','B','V','R','I','J','H','K']:
            phot = 'opt'
            band = '{}mag'.format(b)
        elif b in ['W1','W2','W3','W4']:
            phot = 'ir'
            band = '{}mag'.format(b)
        elif b in ('G'):
            phot = 'gaia'
            band = '{}mag'.format(b)
        elif b in ('BP','RP'):
            phot = 'gaia'
            band = 'G_{}mag'.format(b)

        if phot is None:
            for system, bands in cls.phot_bands.items():
                if b in bands:
                    phot = system
                    band = b
                    break
            if phot is None:
                raise ValueError('Parsec grids cannot resolve band {}!'.format(b))
        return phot, band
        
    @classmethod
    def phot_tarball_file(cls, phot, **kwargs):
        return os.path.join(cls.datadir, '{}.tar.gz'.format(phot))
        
    def get_filenames(self, phot):
        d = os.path.join(self.datadir, '{}'.format(phot))
        if not os.path.exists(d):
            if not os.path.exists(self.phot_tarball_file(phot)):
                self.extract_phot_tarball(phot)

        return [os.path.join(d,f) for f in os.listdir(d) if re.search('\.dat$', f)]

    @classmethod
    def get_feh(cls, filename):
        m = re.search('([mp])([0-9]{3}).', filename)
        if m:
            sign = 1 if m.group(1)=='p' else -1
            return float(m.group(2))/100. * sign
        else:
            raise ValueError('{} not a valid Parsec file? Cannnot parse [Fe/H]'.format(filename))

    @classmethod
    def to_df(cls, filename):
        with open(filename, 'r', encoding='latin-1') as fin:
            while True:
                line = fin.readline()
                if re.match('# Zini', line):
                    column_names = line[1:].split()
                    break
        feh = cls.get_feh(filename)
        df = pd.read_table(filename, comment='#', delim_whitespace=True,
                             skip_blank_lines=True, names=column_names)
        df['feh']=cls.get_feh(filename)
        df['Zini'] = df['feh']#feh
        df['Age'] = np.log10(df['Age'])
        return df

    def df_all(self, phot, **kwargs):
        df = super(ParsecModelGrid, self).df_all(phot)
        df = df.sort_values(by=['feh','Age','Mini'])
        df.index = [df.feh, df.Age]
        return df

    def hdf_filename(self, phot):
        return os.path.join(self.datadir, '{}.h5'.format(phot))
