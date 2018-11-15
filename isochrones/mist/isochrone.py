from ..models import ModelGridInterpolator
from .models import MISTIsochroneGrid, MISTEvolutionTrackGrid
from .bc import MISTBolometricCorrectionGrid


class MISTIsochrone(ModelGridInterpolator):

    grid_type = MISTIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    _param_index_order = (1, 2, 0, 3, 4)


class MISTEvolutionTrack(ModelGridInterpolator):

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    _param_index_order = (2, 0, 1, 3, 4)
