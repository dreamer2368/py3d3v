from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext':build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [Extension("interp",["interp.pyx", "par_interp.cpp"],
                             libraries=["m"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             language="c++"),
                   Extension("ewald",["ewald.pyx"],
                             libraries=["m"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             language="c++"),
                   Extension("core",["core.pyx", "par_core.cpp"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             language="c++"),
                   Extension("solvers",["solvers.pyx", "par_core.cpp"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             language="c++")]
)
