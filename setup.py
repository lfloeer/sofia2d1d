"""
Build configuration of the compiled extension. All package metadata lives in
pyproject.toml.

Environment variables:

SOFIA2D1D_NATIVE=1
    Optimise for the build machine (-march=native). Portable x86-64 Linux
    builds already contain AVX2 versions of the inner loops.
SOFIA2D1D_NO_OPENMP=1
    Build without OpenMP (single-threaded).
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

OPENMP_TEST = r"""
#include <omp.h>
int main(void) {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n > 0 ? 0 : 1;
}
"""


def _flag_set(env, default):
    return os.environ.get(env, default).lower() not in ("", "0", "false", "no")


def _compiles(compiler, compile_args, link_args):
    """Whether a small OpenMP program compiles and links with the given flags."""
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "omp_test.c")
        with open(src, "w") as f:
            f.write(OPENMP_TEST)
        objects = compiler.compile([src], output_dir=tmp, extra_postargs=compile_args)
        compiler.link_executable(objects, "omp_test", output_dir=tmp, extra_postargs=link_args)
        return True
    except Exception:
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _homebrew_libomp():
    """Compile and link flags for Homebrew's libomp on macOS (Apple clang has no OpenMP)."""
    prefixes = []
    try:
        out = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, check=True)
        prefixes.append(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    prefixes += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    for prefix in prefixes:
        if os.path.isdir(os.path.join(prefix, "include")):
            return (
                ["-Xpreprocessor", "-fopenmp", "-I" + os.path.join(prefix, "include")],
                ["-L" + os.path.join(prefix, "lib"), "-lomp"],
            )
    return None


class BuildExt(build_ext):
    def build_extensions(self):
        msvc = self.compiler.compiler_type == "msvc"
        if msvc:
            # /fp:precise does not contract a * b + c into FMA instructions
            compile_args = ["/O2", "/fp:precise"]
            candidates = [(["/openmp"], [])]
        else:
            compile_args = ["-O3", "-ffp-contract=off"]
            candidates = [(["-fopenmp"], ["-fopenmp"])]
            if sys.platform == "darwin":
                libomp = _homebrew_libomp()
                if libomp:
                    candidates.append(libomp)
            if _flag_set("SOFIA2D1D_NATIVE", ""):
                compile_args.append("-march=native")

        openmp = None
        if not _flag_set("SOFIA2D1D_NO_OPENMP", ""):
            for flags in candidates:
                if _compiles(self.compiler, *flags):
                    openmp = flags
                    break
            if openmp is None:
                self.warn("OpenMP not available, sofia2d1d will run on a single thread")

        for ext in self.extensions:
            ext.extra_compile_args = compile_args + (openmp[0] if openmp else [])
            ext.extra_link_args = openmp[1] if openmp else []
        super().build_extensions()


extensions = [
    Extension(
        name="sofia2d1d.pyudwt",
        sources=["src/sofia2d1d/pyudwt.pyx"],
        language="c++",
        include_dirs=[numpy.get_include(), "src/sofia2d1d"],
        depends=["src/sofia2d1d/_kernels.h"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
    cmdclass={"build_ext": BuildExt},
)
