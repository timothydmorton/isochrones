import os
import pickle

from ..isochrone import Isochrone
from ..config import ISOCHRONES
from .grid import YAPSIModelGrid

class YAPSI_Isochrone(Isochrone):
    
    name = 'dartmouth'
    default_bands = YAPSIModelGrid.default_bands
    tri_file = os.path.join(ISOCHRONES, 'yapsi.tri')
    
    def __init__(self, **kwargs):
        df = YAPSIModelGrid().df

        with open(self.tri_file, 'rb') as f:
            tri = pickle.load(f)
        
        mags = {b:df[b].values for b in self.default_bands}

        Isochrone.__init__(self,df['mass'].values, df['age'].values,
                           df['feh'].values,df['mass'].values, df['logL'].values,
                           10**df['logTeff'].values,df['logg'].values,mags,tri=tri, 
                           **kwargs)

        
        