
import numpy as np
cimport numpy as np
import scipy as sp
from libc.math cimport floor, ceil, exp, erf, fabs, sqrt, cos, sin

ctypedef np.float64_t DOUBLE

cdef extern from "par_core.hpp":

    void move_par(const int N, const double dt,
                  double *zp, const double *vz,
                  double *yp, const double *vy,
                  double *xp, const double *vx)

    void accel_par(const int N, const double dt, const double *qm,
                   const double *Ez, double *vz,
                   const double *Ey, double *vy,
                   const double *Ex, double *vx)

    void scale_array(const int N, double *x, double sx)

    void scale_array_3_copy(const int N,
                            const double *z, const double sz, double *zc,
                            const double *y, const double sy, double *yc,
                            const double *x, const double sx, double *xc)

def calc_Ez(phi, dz):
    
    E           = np.zeros_like(phi)
    E[1:-1,:,:] = -(phi[2:,:,:]-phi[:-2,:,:])
    E[0,:,:]    = -(phi[1,:,:]-phi[-1,:,:])
    E[-1,:,:]   = -(phi[0,:,:]-phi[-2,:,:])
    
    return E/(2*dz)

def calc_Ey(phi, dy):
    
    E           = np.zeros_like(phi)
    E[:,1:-1,:] = -(phi[:,2:,:]-phi[:,:-2,:])
    E[:,0,:]    = -(phi[:,1,:]-phi[:,-1,:])
    E[:,-1,:]   = -(phi[:,0,:]-phi[:,-2,:])
    
    return E/(2*dy)

def calc_Ex(phi, dx):
    
    E           = np.zeros_like(phi)
    E[:,:,1:-1] = -(phi[:,:,2:]-phi[:,:,:-2])
    E[:,:,0]    = -(phi[:,:,1]-phi[:,:,-1])
    E[:,:,-1]   = -(phi[:,:,0]-phi[:,:,-2])
    
    return E/(2*dx)


def normalize(double[:] x, double L):
    """ Keep x in [0,L), assuming a periodic domain
    """
    # The order here is significant because of rounding
    # If x<0 is very close to 0, then float(x+L)=L
    #while len(x[x<0])>0 or len(x[x>=L])>0:
    cdef int N = x.shape[0]
    cdef int i
    for i in range(N):
        if x[i]<0:
            x[i] += L
        if x[i] >= L:
            x[i] -= L


def move(double dt,
         double[:] zp, double[:] vz, double Lz,
         double[:] yp, double[:] vy, double Ly,
         double[:] xp, double[:] vx, double Lx):
        """Move in place
        """

        cdef int N = zp.shape[0]
        move_par(N, dt,
                 &zp[0], &vz[0],
                 &yp[0], &vy[0],
                 &xp[0], &vx[0])

        normalize(zp, Lz)
        normalize(yp, Ly)
        normalize(xp, Lx)

def accel(double dt, double[:] qm,
          double[:] Ez, double[:] vz,
          double[:] Ey, double[:] vy,
          double[:] Ex, double[:] vx):

    Ez = np.ascontiguousarray(Ez)
    Ey = np.ascontiguousarray(Ey)
    Ex = np.ascontiguousarray(Ex)
    cdef int N = Ez.shape[0]

    accel_par(N, dt, &qm[0],
              &Ez[0], &vz[0],
              &Ey[0], &vy[0],
              &Ex[0], &vx[0])


def scale(double[:] x, double sx):

    cdef int N = x.shape[0]
    scale_array(N, &x[0], sx)


def scale3copy(double[:] z, double sz, double[:] zc,
               double[:] y, double sy, double[:] yc,
               double[:] x, double sx, double[:] xc):

    cdef int N = z.shape[0]
    scale_array_3_copy(N,
                       &z[0], sz, &zc[0],
                       &y[0], sy, &yc[0],
                       &x[0], sx, &xc[0])
