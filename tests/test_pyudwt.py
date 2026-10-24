import numpy as np
import pytest

from sofia2d1d import pyudwt
from sofia2d1d.sourcefinder import denoise_2d1d

import reference


def make_cube(shape, seed=0, sources=8):
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal(shape).astype(np.float32)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    for _ in range(sources):
        cz, cy, cx = rng.uniform(0, 1, 3) * shape
        w = rng.uniform(1, 4)
        cube += np.float32(rng.uniform(1, 4)) * np.exp(
            -((z - cz)**2 / (2 * (2 * w)**2) + (y - cy)**2 / (2 * w**2) + (x - cx)**2 / (2 * w**2))
        ).astype(np.float32)
    return cube


SHAPES = [(16, 24, 20), (17, 23, 13), (40, 31, 70), (5, 9, 70), (64, 8, 8)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("options", [
    {},
    {'xy_approx': True, 'z_approx': True, 'total_power': True, 'positivity': False, 'iterations': 2},
    {'total_power': True, 'xy_scales': 2, 'z_scales': 3},
    {'z_approx': True, 'xy_scales': 0},
    {'xy_approx': True, 'z_scales': 0, 'iterations': 1},
])
def test_adaptive_matches_reference(shape, options):
    cube = make_cube(shape)
    options = dict(options)
    iterations = options.pop('iterations', 3)
    positivity = options.pop('positivity', True)
    expected, thresholds, mrs = reference.denoise(cube, sigma=3, iterations=iterations,
                                                  positivity=positivity, **options)

    denoiser = pyudwt.Denoise2D1DAdaptiveMRS(3, cube, **options)
    denoiser.positivity = positivity
    for _ in range(iterations):
        denoiser.decompose()

    np.testing.assert_array_equal(denoiser.reconstruction, expected)
    support = denoiser.mrs
    for (xy, z), m in mrs.items():
        np.testing.assert_array_equal(support[xy, z].astype(bool), m)
        assert denoiser.thresholds[xy, z] == thresholds[xy, z]


@pytest.mark.parametrize("mode", ["periodic", "reflect", "zero"])
@pytest.mark.parametrize("cls, rule", [
    (pyudwt.Denoise2D1DHard, 'hard'),
    (pyudwt.Denoise2D1DSoft, 'soft'),
    (pyudwt.Denoise2D1DHardMRS, 'mrs'),
])
def test_fixed_thresholds(mode, cls, rule):
    cube = make_cube((24, 20, 19), seed=1)
    denoiser = cls(cube, total_power=True)
    denoiser.boundary = mode
    t = np.linspace(0.2, 1.5, denoiser.thresholds.size).reshape(denoiser.thresholds.shape).astype(np.float32)
    t[1, 2] = -1  # switched off
    denoiser.thresholds = t
    denoiser.positivity = True
    for _ in range(2):
        denoiser.decompose()
    expected, _, _ = reference.denoise(cube, thresholds=t, rule=rule, iterations=2,
                                       total_power=True, mode=mode)
    np.testing.assert_array_equal(denoiser.reconstruction, expected)


def test_fix_mrs():
    cube = make_cube((32, 30, 28), seed=2)
    denoiser = pyudwt.Denoise2D1DAdaptiveMRS(3, cube)
    denoiser.decompose()
    denoiser.fix_mrs = True
    denoiser.decompose()
    denoiser.decompose()
    expected, _, _ = reference.denoise(cube, sigma=3, fix_mrs_after=1, iterations=3, positivity=False)
    np.testing.assert_array_equal(denoiser.reconstruction, expected)


def test_denoise_2d1d_handles_invalid_values():
    cube = make_cube((32, 33, 34), seed=3)
    cube[3, :4, :5] = np.nan
    cube[10, 7, 7] = np.inf
    result = denoise_2d1d(cube, sigma=3)
    clean = np.where(np.isfinite(cube), cube, 0).astype(np.float32)
    expected, _, _ = reference.denoise(clean, sigma=3)
    expected[~np.isfinite(cube)] = 0
    np.testing.assert_array_equal(result, expected)


def test_results_do_not_depend_on_threads():
    cube = make_cube((48, 64, 72), seed=4)
    n = pyudwt.get_num_threads()
    try:
        results = []
        for threads in (1, 3, max(n, 2)):
            pyudwt.set_num_threads(threads)
            results.append(denoise_2d1d(cube, sigma=3))
    finally:
        pyudwt.set_num_threads(n)
    for r in results[1:]:
        np.testing.assert_array_equal(r, results[0])


class Collector(pyudwt.WaveletDecomposition2D1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bands = []

    def handle_coefficients(self, work_array, xy_scale, z_scale):
        self.bands.append((xy_scale, z_scale, self.work[work_array].copy()))


@pytest.mark.parametrize("shape", SHAPES[:3])
def test_generic_decomposition(shape):
    cube = make_cube(shape, seed=5)
    collector = Collector(cube)
    collector.decompose()
    expected = list(reference.subbands(cube, collector.xy_scales, collector.z_scales))
    assert [b[:2] for b in collector.bands] == [b[:2] for b in expected]
    for (_, _, got), (_, _, want) in zip(collector.bands, expected):
        np.testing.assert_array_equal(got, want)
    # The transform is exactly invertible by summation (up to rounding)
    total = np.sum([b[2].astype(np.float64) for b in collector.bands], axis=0)
    np.testing.assert_allclose(total, cube, atol=1e-4)


def test_3d_decomposition_reconstructs():
    cube = make_cube((20, 24, 22), seed=6)
    bands = []

    class Collector3D(pyudwt.WaveletDecomposition3D):
        def handle_coefficients(self, work_array, scale):
            bands.append(self.work[work_array].astype(np.float64))

    Collector3D(cube).decompose()
    np.testing.assert_allclose(np.sum(bands, axis=0), cube, atol=1e-4)


def test_1d_decomposition():
    rng = np.random.default_rng(7)
    spectrum = rng.standard_normal(256).astype(np.float32)
    spectrum[100:120] += 10
    bands = []

    class Collector1D(pyudwt.WaveletDecomposition1D):
        def handle_coefficients(self, work_array, scale):
            bands.append(self.work[work_array].astype(np.float64))

    Collector1D(spectrum).decompose()
    np.testing.assert_allclose(np.sum(bands, axis=0), spectrum, atol=1e-4)

    denoiser = pyudwt.Denoise1DHardMRS(spectrum, thresholds=[3.] * 8)
    denoiser.decompose()
    denoiser.decompose()
    assert np.all(denoiser.reconstruction[:80] == 0)
    assert denoiser.reconstruction[105:115].mean() > 2


def test_mrs_roundtrip():
    cube = make_cube((16, 12, 21), seed=8)
    denoiser = pyudwt.Denoise2D1DAdaptiveMRS(3, cube)
    denoiser.decompose()
    support = denoiser.mrs
    assert support.shape == (denoiser.xy_scales, denoiser.z_scales) + cube.shape
    assert support.any()
    denoiser.mrs = np.zeros_like(support)
    assert not denoiser.mrs.any()
    denoiser.mrs = support
    np.testing.assert_array_equal(denoiser.mrs, support)
