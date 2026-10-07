"""Dirichlet splatting: coherent Slang rendering and the DSFW inverse solver.

parameters  per-atom parameter constraints
spectral    Dirichlet atoms on a spectral grid
surfel      oriented surfels: parameters, scattering and material, rendering
fmcw        FMCW sensor, scans and the surfel measurement model
groundtruth planar arrays and FMCW scans synthesized from the beat signal
dsfw        the solver loop over varpro, certificate, replacement, refinement, spatial
kernels     Slang kernels and their PyTorch bindings
"""

from .certificate import CertificateSearchConfig
from .dsfw import DSFWConfig, fit, initialize, step
from .fmcw import FMCWConfig, FMCWModel, Scan
from .parameters import ParameterConfig
from .spatial import SpatialSearchConfig
from .spectral import SpectralGrid, SpectralModel

__all__ = [
    "CertificateSearchConfig",
    "DSFWConfig",
    "FMCWConfig",
    "FMCWModel",
    "ParameterConfig",
    "Scan",
    "SpatialSearchConfig",
    "SpectralGrid",
    "SpectralModel",
    "fit",
    "initialize",
    "step",
]
__version__ = "0.1.0"
