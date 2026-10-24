import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

compile_args = ['-O3', '-fopenmp', '-ffp-contract=off']
# Optimise for the build machine (AVX2/AVX-512, FMA). Results may then differ
# from portable builds in the last bits of the floating point values.
if os.environ.get('SOFIA2D1D_NATIVE'):
    compile_args.append('-march=native')

extensions = [
    Extension(
        name="sofia2d1d.pyudwt",
        sources=["sofia2d1d/pyudwt.pyx"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include(), "sofia2d1d"],
        depends=["sofia2d1d/_kernels.h"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


setup(
    name='sofia2d1d',
    author="Lars Floeer",
    ext_modules=cythonize(extensions),
    packages=['sofia2d1d'],
    install_requires=['numpy'],
)
