
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil
import cython

ctypedef np.float64_t DOUBLE

cdef extern from "par_interp.h":
    void interp_cic_par(const int nz, const int ny, const int nx, const double *vals,
                        const int N, const double *z, const double *y, const double *x, double *c)

        
def interp_cic(double[:,:,:] vals,
               double[:] z,
               double[:] y,
               double[:] x):

    cdef np.ndarray[DOUBLE,ndim=1] c = np.zeros_like(z, dtype=np.double)
    cdef int nz, ny, nx
    nz, ny, nx = vals.shape[0], vals.shape[1], vals.shape[2] 
    cdef int N = x.shape[0]
    cdef double[:,:,:] vd = np.ascontiguousarray(vals)
    cdef double[:]     zd = np.ascontiguousarray(z)
    cdef double[:]     yd = np.ascontiguousarray(y)
    cdef double[:]     xd = np.ascontiguousarray(x)
    cdef double[:]     cd = np.ascontiguousarray(c)

    interp_cic_par(nz, ny, nx, &vd[0,0,0],
                   N, &zd[0], &yd[0], &x[0], &cd[0])

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
