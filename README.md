sofia2d1d
=========

Implementation of 2D-1D Wavelet De-noising for SoFiA.

The algorithm is based on 
```
Starck et al. 2009
Source detection using a 3D sparse representation: application to the Fermi gamma-ray space telescope
DOI 10.1051/0004-6361/200811388
```

Applications to radio-astronmical data are discussed in
```
Flöer & Winkel 2012
2D-1D Wavelet Reconstruction as a Tool for Source Finding in Spectroscopic Imaging Surveys
DOI 10.1071/AS11042
```
```
Flöer et al. 2014
Source finding, parametrization, and classification for the extragalactic Effelsberg-Bonn HI Survey
DOI 10.1051/0004-6361/201424415
```

Installation
------------

```
pip install git+https://github.com/lfloeer/sofia2d1d
```

(or `uv add git+https://github.com/lfloeer/sofia2d1d`). Installing from
source needs a C++ compiler; Python >= 3.10 and NumPy >= 1.25 are supported.
OpenMP is used when the compiler supports it (on macOS: `brew install libomp`);
without it, the package works but runs on a single thread
(`sofia2d1d.openmp_enabled()` tells which). Build options, as environment
variables:

* `SOFIA2D1D_NATIVE=1`: optimise for the build machine. Portable builds on
  x86-64 Linux already contain AVX2 versions of the inner loops that are picked
  at run time. Results are identical either way.
* `SOFIA2D1D_NO_OPENMP=1`: build without OpenMP.

Usage
-----

```python
import sofia2d1d

sofia2d1d.set_num_threads(8)  # default: OMP_NUM_THREADS or all cores
reconstruction = sofia2d1d.denoise_2d1d(cube, sigma=5, iterations=3)
```

The first axis of the cube is the spectral axis. Invalid values (NaN, inf or
those flagged by `valid=`) are set to zero before and after the
reconstruction, which is non-negative unless `positivity=False`.

Development
-----------

The project is managed with [uv](https://docs.astral.sh/uv/):

```
uv sync                  # create .venv, build the extension, install dev tools
uv run pytest            # tests (exact comparison with tests/reference.py)
uv run ruff check .      # lint
uv run ruff format .     # format
uv build                 # sdist and wheel in dist/
```

uv rebuilds the extension automatically when `pyudwt.pyx`, `_kernels.h`,
`setup.py` or `pyproject.toml` change. Wheels for Linux, macOS and Windows are
built with cibuildwheel by the `Wheels` workflow when a `v*` tag is pushed,
and published to PyPI once trusted publishing is set up for the project.

Layout:

* `src/sofia2d1d/pyudwt.pyx`: the transform and de-noising classes (Cython)
* `src/sofia2d1d/_kernels.h`: vectorised inner loops (C)
* `src/sofia2d1d/sourcefinder.py`: `denoise_2d1d`, the high-level interface
* `tests/reference.py`: plain NumPy implementation used as the test oracle

Performance
-----------

The transform is parallelised with OpenMP and organised to keep memory traffic
and memory use low:

* The spectral part of the transform runs on cache-sized tiles of spectra:
  after the spatial smoothing of a spatial scale, all spectral sub-bands of a
  tile are computed and thresholded in cache, so no sub-band is ever stored as
  a full cube.
* The noise level of each sub-band (median absolute coefficient) is computed
  exactly with a histogram of the leading bits of the coefficients and a
  selection among the few coefficients in the bin of the median, i.e. without
  copying or sorting the sub-band.
* The multi-resolution support takes one bit per coefficient.

Peak memory is about 16 bytes per voxel plus one bit per voxel and sub-band
(e.g. 9 bytes per voxel for 9 x 8 sub-bands), compared to roughly 150-200 bytes
per voxel before. Reconstructions are bit-for-bit identical to the original
implementation; `tests/reference.py` is a plain NumPy version of the algorithm
that the test suite (`uv run pytest`) compares against.

Timings for `denoise_2d1d` with default parameters on 4 cores:

| cube           | original | now    |
|----------------|---------:|-------:|
| 64 x 96 x 80   | 1.4 s    | 0.15 s |
| 128 x 128 x 128| 7.4 s    | 0.48 s |
| 256 x 256 x 256| 98 s     | 3.8 s  |
| 256 x 512 x 512| -        | 18 s   |
