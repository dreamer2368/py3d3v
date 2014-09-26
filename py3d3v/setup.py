from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext':build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [Extension("interp",["interp.pyx", "par_interp.c"],
                             libraries=["m"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp']),
                   Extension("tools",["tools.pyx", "par_tools.c"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'])]
    )
