from ..models import ModelGridInterpolator
from .models import MISTIsochroneGrid, MISTEvolutionTrackGrid
from .bc import MISTBolometricCorrectionGrid


class MIST_Isochrone(ModelGridInterpolator):

    grid_type = MISTIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    _param_index_order = (1, 2, 0, 3, 4)
    param_names = ('eep', 'age', 'feh', 'distance', 'AV')
    eep_replaces = 'mass'
    eep_bounds = (0, 1710)

class MIST_EvolutionTrack(ModelGridInterpolator):

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    _param_index_order = (2, 0, 1, 3, 4)
    param_names = ('mass', 'eep', 'feh', 'distance', 'AV')
    eep_replaces = 'age'
    eep_bounds = (0, 1710)
