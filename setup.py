"""
Build configuration of the compiled extension. All package metadata lives in
pyproject.toml.

Environment variables:

SOFIA2D1D_NATIVE=1
    Optimise for the build machine (-march=native). Portable x86-64 Linux
    builds already contain AVX2 versions of the inner loops.
SOFIA2D1D_NO_OPENMP=1
    Build without OpenMP (single-threaded).
SOFIA2D1D_LIBOMP_PREFIX=/path/to/prefix
    Use the OpenMP runtime (libomp) installed in this prefix, e.g. a
    conda-forge llvm-openmp environment. Needed with compilers that do not ship
    an OpenMP runtime (Apple clang) unless Homebrew's libomp is found.
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


def _libomp_flags(prefix):
    """Compile and link flags for the libomp in prefix (for clang without its own OpenMP runtime)."""
    lib = os.path.join(prefix, "lib")
    return (
        ["-Xpreprocessor", "-fopenmp", "-I" + os.path.join(prefix, "include")],
        ["-L" + lib, "-lomp", "-Wl,-rpath," + lib],
    )


def _libomp_prefixes():
    """Candidate libomp installations: SOFIA2D1D_LIBOMP_PREFIX, then Homebrew on macOS."""
    prefixes = []
    if os.environ.get("SOFIA2D1D_LIBOMP_PREFIX"):
        prefixes.append(os.environ["SOFIA2D1D_LIBOMP_PREFIX"])
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, check=True)
            prefixes.append(out.stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            pass
        prefixes += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    return [p for p in prefixes if os.path.isfile(os.path.join(p, "include", "omp.h"))]


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
            libomp = [_libomp_flags(p) for p in _libomp_prefixes()]
            # An explicitly requested libomp takes precedence over the compiler's own runtime
            if os.environ.get("SOFIA2D1D_LIBOMP_PREFIX"):
                candidates = libomp + candidates
            else:
                candidates += libomp
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
