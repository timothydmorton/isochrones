import pandas as pd
import numpy as np

try:
    from astropy import constants as const

    #Define useful constants
    G = const.G.cgs.value
    MSUN = const.M_sun.cgs.value
    RSUN = const.R_sun.cgs.value
except ImportError:
    G = 6.67e-11
    MSUN = 1.99e33
    RSUN = 6.96e10


from ..extinction import EXTINCTION, LAMBDA_EFF, extcurve, extcurve_0
from ..isochrone import Isochrone
from .utils import interp_value, interp_values

class MagFunction(object):
    def __init__(self, ic, band, icol):
        self.ic = ic
        self.band = band
        self.icol = icol

    def __call__(self, mass, age, feh):
        return self.ic.interp_value(mass, age, feh, self.icol)

class MIST_Isochrone(Isochrone):
    
    def __init__(self, df, ext_table=False):
        # df should be indexed by [feh, age]
        self.df = df

        self.ext_table = ext_table

        self.mass_col = 2
        self.Ncols = df.shape[1]
    
        self.fehs = self.df.feh.unique()
        self.ages = self.df.log10_isochrone_age_yr.unique()
        self.Nfeh = len(self.fehs)
        self.Nage = len(self.ages)
    
        #organized array
        self._grid = None
        self._grid_Ns = None
        
        self._props = ['mass', 'logTeff', 'logg', 'logL', 'Z_surf']
        self._prop_cols = [1, 2, 3, 4, 5, 6]

        self.bands = ['u','g','r','i','z']
        self._mag_cols = {'u':7, 'g':8, 'r':9, 'i':10, 'z':11}
        self._mag = {b: MagFunction(self, b, i)
                            for b,i in self._mag_cols.items()}
        self.mag = {b:self._mag_fn(b) for b in self.bands}

    
        self.minage = self.ages.min()
        self.maxage = self.ages.max()
        self.minmass = self.df.iloc[:, self.mass_col].min()
        self.maxmass = self.df.iloc[:, self.mass_col].max()
        self.minfeh = self.fehs.min()
        self.maxfeh = self.fehs.max()

    def logTeff(self, mass, age, feh):
        return self.interp_value(mass, age, feh, 3)

    def logg(self, mass, age, feh):
        return self.interp_value(mass, age, feh, 4)

    def logL(self, mass, age, feh):
        return self.interp_value(mass, age, feh, 5)

    def Z_surf(self, mass, age, feh):
        return self.interp_value(mass, age, feh, 6)

    def radius(self, *args):
        return np.sqrt(G*self.mass(*args)*MSUN/10**self.logg(*args))/RSUN

    def Teff(self, *args):
        return 10**self.logTeff(*args)

    def mass(self, *args):
        return args[0]

    @property
    def grid(self):
        if self._grid is None:
            self._make_grid()
        return self._grid
    
    @property
    def grid_Ns(self):
        if self._grid_Ns is None:
            self._make_grid()
        return self._grid_Ns
        
    def _make_grid(self):
        df_list = [[self.df.ix[f,a] for f in self.fehs] for a in self.ages]
        lens = np.array([[len(df_list[i][j]) for j in range(self.Nfeh)] 
                         for i in range(self.Nage)]).T #just because
        data = np.zeros((self.Nfeh, self.Nage, lens.max(), self.Ncols))

        for i in range(self.Nage):
            for j in range(self.Nfeh):
                N = lens[j,i]
                data[j, i, :N, :] = df_list[i][j].values
                data[j, i, N:, :] = np.nan
        
        self._grid = data
        self._grid_Ns = lens
                
    def interp_value(self, mass, age, feh, icol): # 4 is log_g
        try:
            return interp_value(mass, age, feh, icol,
                                self.grid, self.mass_col,
                                self.ages, self.fehs, self.grid_Ns)

        except:
            # First, broadcast to common shape.
            b = np.broadcast(mass, age, feh)
            mass = np.resize(mass, b.shape)
            age = np.resize(age, b.shape)
            feh = np.resize(feh, b.shape)

            # Then pass to helper function
            return interp_values(mass, age, feh, icol,
                                self.grid, self.mass_col,
                                self.ages, self.fehs, self.grid_Ns)
