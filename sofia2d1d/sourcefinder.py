import numpy as np
from scipy import ndimage as nd

from .pyudwt import Denoise2D1DHardMRS

b3spline = np.array([1.,4.,6.,4.,1.]) / 16.

# Try to use MUCH faster median implementation
# from bottleck, else fallback to numpy.median
try:
    from bottleneck import median as the_median
except ImportError:
    the_median = np.median


class Denoise2D1DAdaptiveMRS(Denoise2D1DHardMRS):
    """
    De-noise a three-dimensional data cube using the 2D1D wavelet
    transform with a multi-resolution support as discussed in 
    Starck et al. 2011 and Flöer & Winkel 2012.

    The transform assumes that the first axis is the spectral axis.
    """

    def __init__(self, sigma, *args, **kwargs):

        self.sigma = sigma

        self.xy_mother_function = b3spline
        self.z_mother_function = b3spline

        #TODO: Bad Hack
        kwargs['thresholds'] = np.zeros((1, 1, 1, 1))
        
        super(Denoise2D1DAdaptiveMRS, self).__init__(*args, **kwargs)
        
        #TODO: Bad Hack
        self.thresholds = np.zeros((1, 1, self.xy_scales + 1, self.z_scales + 1))

    
    def handle_coefficients(self, work_array, xy_scale, z_scale):
        
        subband_rms = the_median(np.abs(self.work[work_array])) / 0.6745
        subband_threshold = subband_rms * self.sigma
        
        #TODO: Bad Hack
        self.thresholds[0, 0, xy_scale, z_scale] = subband_threshold
        
        super(Denoise2D1DAdaptiveMRS, self).handle_coefficients(work_array, xy_scale, z_scale)


def denoise_2d1d(data, sigma=5, xy_scales=-1, z_scales=-1, iterations=3, valid=None, **kwargs):
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

    iterations : int, optional
        The number of iterations for the reconstruction process.

    valid : 3d ndarray of bool, optional
        Mask indicating the valid values in the data. True means good data.
        False values are set to 0. prior to reconstruction.
        Gets deduced from the data if not provided.

    Other arguments are passed to the denoising class.


    Returns
    -------

    reconstruction : 3d ndarray
        The reconstructed data
    """

    data = np.array(data, dtype=np.single)
    
    if valid is None:
        valid = np.isfinite(data)

    data[~valid] = 0.

    denoiser = Denoise2D1DAdaptiveMRS(
        sigma=sigma,
        data=data,
        xy_scales=xy_scales,
        z_scales=z_scales,
        **kwargs)

    for iteration in xrange(iterations):
        denoiser.decompose()

    reconstruction = denoiser.reconstruction
    reconstruction[~valid] = 0.

    return reconstruction
