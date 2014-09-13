
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil

cpdef build_k2(int nz, double dz,
               int ny, double dy,
               int nx, double dx):

    cdef np.ndarray kz = np.fft.fftfreq(nz)*2*np.pi/dz
    cdef np.ndarray ky = np.fft.fftfreq(ny)*2*np.pi/dy
    cdef np.ndarray kx = np.fft.fftfreq(nx)*2*np.pi/dx

    cdef int nkz, nky, nkx
    cdef int iz, iy, ix
    
    nkz, nky, nkx = len(kz), len(ky), len(kx)
    cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)
    for iz in range(nkz):
        for iy in range(nky):
            for ix in range(nkx):
                k2_vals[iz,iy,ix] = kz[iz]**2+ky[iy]**2+kx[ix]**2
    # Avoid a divide by zero            
    k2_vals[0,0,0] = 1.

    return k2_vals
