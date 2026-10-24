import numpy as np

from .pyudwt import Denoise2D1DAdaptiveMRS, b3spline, set_num_threads, get_num_threads


def denoise_2d1d(data, sigma=5, xy_scales=-1, z_scales=-1, positivity=True, iterations=3, valid=None, **kwargs):
    """

    Inputs
    ------

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

    Other arguments are passed to the denoising class. Possible arguments are:
        xy_approx : bool
            Whether to consider the wavelet sub-bands representing the spatial
            approximation during reconstruction

        z_approx : bool
            Whether to consider the wavelet sub-bands representing the spectral
            approximation during reconstruction

        total_power : bool
            Whether to add the smooth approximation to the reconstruction

    The number of threads is taken from OMP_NUM_THREADS or can be set with
    sofia2d1d.sourcefinder.set_num_threads.


    Returns
    -------

    reconstruction : 3d ndarray
        The reconstructed data
    """

    data = np.array(data, dtype=np.single, order='C')

    if valid is None:
        valid = np.isfinite(data)
    invalid = ~np.asarray(valid, dtype=bool)

    data[invalid] = 0.

    denoiser = Denoise2D1DAdaptiveMRS(
        sigma=sigma,
        data=data,
        xy_scales=xy_scales,
        z_scales=z_scales,
        **kwargs)

    denoiser.positivity = positivity

    for iteration in range(iterations):
        denoiser.decompose()

    reconstruction = denoiser.reconstruction
    reconstruction[invalid] = 0.

    return reconstruction
