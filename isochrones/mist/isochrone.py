from ..models import EvolutionTrackInterpolator, IsochroneInterpolator
from .models import MISTIsochroneGrid, MISTBasicIsochroneGrid, MISTEvolutionTrackGrid
from .bc import MISTBolometricCorrectionGrid

class MIST_Isochrone(IsochroneInterpolator):
    grid_type = MISTIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


class MIST_BasicIsochrone(IsochroneInterpolator):
    grid_type = MISTBasicIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


class MIST_EvolutionTrack(EvolutionTrackInterpolator):
    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


class MIST_BasicEvolutionTrack(EvolutionTrackInterpolator):
    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


MIST_Isochrone._track_type = MIST_EvolutionTrack
MIST_BasicIsochrone._track_type = MIST_BasicEvolutionTrack
MIST_EvolutionTrack._iso_type = MIST_Isochrone
MIST_BasicEvolutionTrack._iso_type = MIST_BasicIsochrone
