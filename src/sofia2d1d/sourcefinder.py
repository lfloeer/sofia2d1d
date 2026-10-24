"""High-level interface to the 2D-1D wavelet de-noising."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pyudwt import Denoise2D1DAdaptiveMRS, b3spline, get_num_threads, openmp_enabled, set_num_threads

__all__ = ["denoise_2d1d", "b3spline", "get_num_threads", "openmp_enabled", "set_num_threads"]


def denoise_2d1d(
    data: ArrayLike,
    sigma: float = 5,
    xy_scales: int = -1,
    z_scales: int = -1,
    positivity: bool = True,
    iterations: int = 3,
    valid: ArrayLike | None = None,
    *,
    xy_approx: bool = False,
    z_approx: bool = False,
    total_power: bool = False,
) -> NDArray[np.float32]:
    """
    De-noise a spectral cube with the iterative 2D-1D wavelet reconstruction
    using an adaptive multi-resolution support.

    Parameters
    ----------

    data : 3d ndarray
        The data to denoise. The data already has to have all weights applied to
        it.

    sigma : float, optional
        The reconstruction threshold. Gets multiplied by the noise in each
        wavelet sub-band.

    xy_scales : int, optional
        The number of spatial scales to use for decomposition.

    z_scales : int, optional
        The number of spectral scales to use for decomposition.

    positivity : bool, optional
        Enforce positivity on the reconstruction

    iterations : int, optional
        The number of iterations for the reconstruction process.

    valid : 3d ndarray of bool, optional
        Mask indicating the valid values in the data. True means good data.
        False values are set to 0. prior to reconstruction.
        Gets deduced from the data if not provided.

    xy_approx : bool, optional
        Whether to consider the wavelet sub-bands representing the spatial
        approximation during reconstruction

    z_approx : bool, optional
        Whether to consider the wavelet sub-bands representing the spectral
        approximation during reconstruction

    total_power : bool, optional
        Whether to add the smooth approximation to the reconstruction

    The number of threads is taken from OMP_NUM_THREADS or can be set with
    sofia2d1d.set_num_threads.

    Returns
    -------

    reconstruction : 3d ndarray
        The reconstructed data
    """

    data = np.array(data, dtype=np.single, order="C")
    if data.ndim != 3:
        raise ValueError("data must be a three-dimensional cube (spectral axis first)")

    if valid is None:
        valid = np.isfinite(data)
    invalid = ~np.asarray(valid, dtype=bool)
    if invalid.shape != data.shape:
        raise ValueError("valid must have the same shape as data")

    data[invalid] = 0.0

    denoiser = Denoise2D1DAdaptiveMRS(
        sigma=sigma,
        data=data,
        xy_scales=xy_scales,
        z_scales=z_scales,
        xy_approx=xy_approx,
        z_approx=z_approx,
        total_power=total_power,
    )
    denoiser.positivity = positivity

    for _ in range(iterations):
        denoiser.decompose()

    reconstruction = denoiser.reconstruction
    reconstruction[invalid] = 0.0

    return reconstruction
