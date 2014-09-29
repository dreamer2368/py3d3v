
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil
import cython

ctypedef np.float64_t DOUBLE

cdef extern from "par_interp.h":
    void interp_cic_par(const int nz, const int ny, const int nx, const double *vals,
                        const int N, const double *z, const double dz,
                        const double *y, const double dy,
                        const double *x, const double dx, double *c)

    void weight_cic_par(const int nz, const int ny, const int nx, double *grid,
                        const int N, const double *z, const double dz, const double *y, const double dy,
                        const double *x, const double dx, const double *q)

        
def interp_cic(double[:,:,:] vals,
               double[:] z, double dz,
               double[:] y, double dy, 
               double[:] x, double dx):

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
                   N, &zd[0], dz, &yd[0], dy, &x[0], dx, &cd[0])

    return c


def weight_cic(double[:,:,:] grid,
               double[:] z, double dz,
               double[:] y, double dy,
               double[:] x, double dx,
               double[:] q,
               double    rho0=0.):
    """Assume all inputs are scaled to grid points
    
    Periodic in x, y, z
    """

    grid[:,:,:] = rho0
    cdef int nz, ny, nx, N
    nz, ny, nx = grid.shape[0], grid.shape[1], grid.shape[2]
    N = z.shape[0]
    cdef double[:,:,:] gd = np.ascontiguousarray(grid)
    cdef double[:]     zd = np.ascontiguousarray(z)
    cdef double[:]     yd = np.ascontiguousarray(y)
    cdef double[:]     xd = np.ascontiguousarray(x)
    cdef double[:]     qd = np.ascontiguousarray(q)

    weight_cic_par(nz, ny, nx, &gd[0,0,0],
                   N, &zd[0], dz, &yd[0], dy, &xd[0], dx, &qd[0])

def weight_cic_1d(double[:] grid,
                  double[:] x, double dx,
                  double[:] q,
                  double rho0=0.):
    """ Weighting to grid (CIC)
    """
    cdef int N  = x.shape[0]
    cdef int nx = grid.shape[0]
    cdef double xis
    cdef int i, left, right
    grid[:] = rho0
    for i in range(N):
        xis   = x[i]/dx
        left  = int(floor(xis))
        right = (left+1)%nx
        grid[left]  += q[i]*(left+1-xis)
        grid[right] += q[i]*(xis-left)
        
    
