
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil

ctypedef np.float64_t DOUBLE

cpdef np.ndarray[DOUBLE,ndim=1] interp_cic(np.ndarray[DOUBLE,ndim=3] vals, 
                                           np.ndarray[DOUBLE,ndim=1] z, 
                                           np.ndarray[DOUBLE,ndim=1] y, 
                                           np.ndarray[DOUBLE,ndim=1] x):
    """Assume all inputs are scaled to grid points
    
    Periodic in x, y, z
    """
    
    cdef int i
    cdef int N = z.shape[0]
    cdef int nz, ny, nx
    cdef double xd, yd, zd
    cdef double x0, x1, y0, y1, z0, z1
    nz, ny, nx = vals.shape[0], vals.shape[1], vals.shape[2]
    
    cdef np.ndarray c = np.zeros(N, dtype=np.float64)
    for i in range(N):
        x0, x1 = floor(x[i]), ceil(x[i])%nx
        y0, y1 = floor(y[i]), ceil(y[i])%ny
        z0, z1 = floor(z[i]), ceil(z[i])%nz
        xd, yd, zd = (x[i]-x0), (y[i]-y0), (z[i]-z0)

        c00  = vals[z0, y0, x0]*(1.-xd)+vals[z0, y0, x1]*xd
        c10  = vals[z0, y1, x0]*(1.-xd)+vals[z0, y1, x1]*xd
        c01  = vals[z1, y0, x0]*(1.-xd)+vals[z1, y0, x1]*xd
        c11  = vals[z1, y1, x0]*(1.-xd)+vals[z1, y1, x1]*xd

        c0   = c00*(1.-yd) + c10*yd
        c1   = c01*(1.-yd) + c11*yd

        c[i] = c0*(1.-zd) + c1*zd

    return c


cpdef weight_cic(np.ndarray[DOUBLE,ndim=3] grid,
                 np.ndarray[DOUBLE,ndim=1] z,
                 np.ndarray[DOUBLE,ndim=1] y,
                 np.ndarray[DOUBLE,ndim=1] x,
                 np.ndarray[DOUBLE,ndim=1] q,
                 double rho0=0.):
    """Assume all inputs are scaled to grid points
    
    Periodic in x, y, z
    """
    cdef int i
    cdef int N = z.shape[0]
    cdef int nz, ny, nx
    cdef double xd, yd, zd, qi
    cdef double x0, x1, y0, y1, z0, z1
    nz, ny, nx = grid.shape[0], grid.shape[1], grid.shape[2]
    
    grid[:,:,:] = rho0
    for i in range(N):
        x0, x1 = floor(x[i]), ceil(x[i])%nx
        y0, y1 = floor(y[i]), ceil(y[i])%ny
        z0, z1 = floor(z[i]), ceil(z[i])%nz
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
