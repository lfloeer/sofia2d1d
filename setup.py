from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        name="sofia2d1d.pyudwt",
        sources=["sofia2d1d/pyudwt.pyx"],
        extra_compile_args=['-O3','-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs = [numpy.get_include()]
    )
]


setup(
    name = 'sofia2d1d',
    author="Lars Floeer",
    ext_modules = extensions,
    cmdclass={'build_ext' : build_ext},
    packages=['sofia2d1d'],
)
