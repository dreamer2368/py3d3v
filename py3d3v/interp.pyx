
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil
import cython

ctypedef np.float64_t DOUBLE

# cpdef np.ndarray[DOUBLE,ndim=1] interp_cic(np.ndarray[DOUBLE,ndim=3] vals, 
#                                            np.ndarray[DOUBLE,ndim=1] z, 
#                                            np.ndarray[DOUBLE,ndim=1] y, 
#                                            np.ndarray[DOUBLE,ndim=1] x):
#     """Assume all inputs are scaled to grid points
    
#     Periodic in x, y, z
#     """
    
#     cdef int i
#     cdef int N = z.shape[0]
#     cdef int nz, ny, nx
#     cdef double xd, yd, zd
#     cdef int x0, x1, y0, y1, z0, z1
#     cdef double c00, c10, c01, c11, c0, c1
#     cdef double xd1m
#     nz, ny, nx = vals.shape[0], vals.shape[1], vals.shape[2]
    
#     cdef np.ndarray[DOUBLE,ndim=1] c = np.zeros(N, dtype=np.float64)
#     for i in range(N):
#         x0, x1 = int(floor(x[i])), int(ceil(x[i])%nx)
#         y0, y1 = int(floor(y[i])), int(ceil(y[i])%ny)
#         z0, z1 = int(floor(z[i])), int(ceil(z[i])%nz)
#         xd, yd, zd = (x[i]-x0), (y[i]-y0), (z[i]-z0)
#         xd1m = 1.-xd

#         c00  = vals[z0, y0, x0]*xd1m+vals[z0, y0, x1]*xd
#         c10  = vals[z0, y1, x0]*xd1m+vals[z0, y1, x1]*xd
#         c01  = vals[z1, y0, x0]*xd1m+vals[z1, y0, x1]*xd
#         c11  = vals[z1, y1, x0]*xd1m+vals[z1, y1, x1]*xd

#         c0   = c00*(1.-yd) + c10*yd
#         c1   = c01*(1.-yd) + c11*yd

#         c[i] = c0*(1.-zd) + c1*zd

#     return c

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void interp_cic_par(double[:,:,:] vals, 
                         double[:] z, 
                         double[:] y,
                         double[:] x,
                         double[:] c) nogil:
    
    cdef int i
    cdef int N = z.shape[0]
    cdef int nz, ny, nx
    cdef double xd, yd, zd
    cdef double c00, c01, c10, c11
    cdef double c0, c1
    cdef int x0, x1, y0, y1, z0, z1
    cdef double xd1m
    nz, ny, nx = vals.shape[0], vals.shape[1], vals.shape[2]
    
    # Not getting better results with prange
    # Revisit once the problem size gets much larger
    for i in range(N):
        x0, x1 = int(floor(x[i])), int(ceil(x[i])%nx)
        y0, y1 = int(floor(y[i])), int(ceil(y[i])%ny)
        z0, z1 = int(floor(z[i])), int(ceil(z[i])%nz)
        xd, yd, zd = (x[i]-x0), (y[i]-y0), (z[i]-z0)
        xd1m = 1.-xd

        c00  = vals[z0, y0, x0]*xd1m+vals[z0, y0, x1]*xd
        c10  = vals[z0, y1, x0]*xd1m+vals[z0, y1, x1]*xd
        c01  = vals[z1, y0, x0]*xd1m+vals[z1, y0, x1]*xd
        c11  = vals[z1, y1, x0]*xd1m+vals[z1, y1, x1]*xd

        c0   = c00*(1.-yd) + c10*yd
        c1   = c01*(1.-yd) + c11*yd

        c[i] = c0*(1.-zd) + c1*zd

        
def interp_cic(vals, z, y, x):
    c = np.zeros_like(z)
    interp_cic_par(vals, z, y, x, c)
    return c

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void weight_cic_par(double[:,:,:] grid,
                         double[:] z,
                         double[:] y,
                         double[:] x,
                         double[:] q,
                         double rho0) nogil:

    cdef int i
    cdef int N = z.shape[0]
    cdef int nz, ny, nx
    cdef double xd, yd, zd, qi
    cdef int x0, x1, y0, y1, z0, z1
    nz, ny, nx = grid.shape[0], grid.shape[1], grid.shape[2]
    
    grid[:,:,:] = rho0
    for i in range(N):
        x0, x1 = int(floor(x[i])), int(ceil(x[i])%nx)
        y0, y1 = int(floor(y[i])), int(ceil(y[i])%ny)
        z0, z1 = int(floor(z[i])), int(ceil(z[i])%nz)
        xd, yd, zd = (x[i]-x0), (y[i]-y0), (z[i]-z0)
        qi = q[i]
        
        grid[z0, y0, x0] += qi*(1-xd)*(1-yd)*(1-zd)
        grid[z0, y0, x1] += qi*xd*(1-yd)*(1-zd)
        grid[z0, y1, x0] += qi*(1-xd)*yd*(1-zd)
        grid[z0, y1, x1] += qi*xd*yd*(1-zd)
        grid[z1, y0, x0] += qi*(1-xd)*(1-yd)*zd
        grid[z1, y0, x1] += qi*xd*(1-yd)*zd
        grid[z1, y1, x0] += qi*(1-xd)*yd*zd
        grid[z1, y1, x1] += qi*xd*yd*zd    


def weight_cic(grid, z, y, x, q, rho0=0.):
    """Assume all inputs are scaled to grid points
    
    Periodic in x, y, z
    """
    weight_cic_par(grid, z, y, x, q, rho0)
