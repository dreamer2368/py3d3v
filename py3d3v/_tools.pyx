
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil

cpdef build_k2(int nz, double dz,
               int ny, double dy,
               int nx, double dx):

    cdef np.ndarray kz2 = (np.fft.fftfreq(nz)*2*np.pi/dz)**2
    cdef np.ndarray ky2 = (np.fft.fftfreq(ny)*2*np.pi/dy)**2
    cdef np.ndarray kx2 = (np.fft.fftfreq(nx)*2*np.pi/dx)**2
    cdef double kz2i, ky2i

    cdef int nkz, nky, nkx
    cdef int iz, iy, ix
    
    nkz, nky, nkx = len(kz2), len(ky2), len(kx2)
    cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)
    for iz in range(nkz):
        kz2i = kz2[iz]
        for iy in range(nky):
            ky2i = ky2[iy]
            for ix in range(nkx):
                k2_vals[iz,iy,ix] = kz2i+ky2i+kx2[ix]
    # Avoid a divide by zero            
    k2_vals[0,0,0] = 1.

    return k2_vals
