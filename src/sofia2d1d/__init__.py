"""
sofia2d1d: fast 2D-1D wavelet de-noising of spectral data cubes.

The cubes are expected in (spectral, y, x) order.
"""

from importlib.metadata import PackageNotFoundError, version

from .pyudwt import (
    Denoise1DHardMRS,
    Denoise2D1DAdaptiveMRS,
    Denoise2D1DHard,
    Denoise2D1DHardMRS,
    Denoise2D1DSoft,
    WaveletDecomposition1D,
    WaveletDecomposition2D1D,
    WaveletDecomposition3D,
    b3spline,
    get_num_threads,
    openmp_enabled,
    set_num_threads,
)
from .sourcefinder import denoise_2d1d

try:
    __version__ = version("sofia2d1d")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "Denoise1DHardMRS",
    "Denoise2D1DAdaptiveMRS",
    "Denoise2D1DHard",
    "Denoise2D1DHardMRS",
    "Denoise2D1DSoft",
    "WaveletDecomposition1D",
    "WaveletDecomposition2D1D",
    "WaveletDecomposition3D",
    "__version__",
    "b3spline",
    "denoise_2d1d",
    "get_num_threads",
    "openmp_enabled",
    "set_num_threads",
]
