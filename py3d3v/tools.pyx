
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil

ctypedef np.float64_t DOUBLE

cdef extern from "par_tools.h":

    void move_par(const int N, const double dt,
                  double *zp, const double *vz,
                  double *yp, const double *vy,
                  double *xp, const double *vx)

    void accel_par(const int N, const double dt, const double *qm,
                   const double *Ez, double *vz,
                   const double *Ey, double *vy,
                   const double *Ex, double *vx)


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

        


# cpdef build_k2(int nz, double dz,
#                int ny, double dy,
#                int nx, double dx):
#     """k2 for CIC weighting
#     """
#     cdef np.ndarray kz = (np.fft.fftfreq(nz)*2*np.pi/dz)
#     cdef np.ndarray ky = (np.fft.fftfreq(ny)*2*np.pi/dy)
#     cdef np.ndarray kx = (np.fft.fftfreq(nx)*2*np.pi/dx)

#     cdef np.ndarray kz2 = (kz)**2
#     cdef np.ndarray ky2 = (ky)**2
#     cdef np.ndarray kx2 = (kx)**2
#     cdef double kz2i, ky2i

#     cdef int nkz, nky, nkx
#     cdef int iz, iy, ix
    
#     nkz, nky, nkx = len(kz2), len(ky2), len(kx2)
#     cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)
#     for iz in range(nkz):
#         kz2i = kz2[iz]
#         for iy in range(nky):
#             ky2i = ky2[iy]
#             for ix in range(nkx):
#                 k2_vals[iz,iy,ix] = (kz2i+ky2i+kx2[ix])
#     # Avoid a divide by zero            
#     k2_vals[0,0,0] = 1.

#     return k2_vals

cpdef build_k2(int nz, double dz,
               int ny, double dy,
               int nx, double dx):
    """k2 for CIC weighting
    """
    cdef np.ndarray kz = (np.fft.fftfreq(nz)*2*np.pi/dz)
    cdef np.ndarray ky = (np.fft.fftfreq(ny)*2*np.pi/dy)
    cdef np.ndarray kx = (np.fft.fftfreq(nx)*2*np.pi/dx)

    cdef np.ndarray skz2 = np.sin(kz*dz/2.)**2
    cdef np.ndarray sky2 = np.sin(ky*dy/2.)**2
    cdef np.ndarray skx2 = np.sin(kx*dx/2.)**2
    cdef double skz2i, skzy2i
    skz2[1:] = skz2[1:]/(kz[1:]*dz/2.)**2
    skz2[0] = 1.
    sky2[1:] = sky2[1:]/(ky[1:]*dy/2.)**2
    sky2[0] = 1.
    skx2[1:] = skx2[1:]/(kx[1:]*dx/2.)**2
    skx2[0] = 1.
    
    cdef np.ndarray kz2 = (kz)**2
    cdef np.ndarray ky2 = (ky)**2
    cdef np.ndarray kx2 = (kx)**2
    cdef double kz2i, ky2i

    cdef int nkz, nky, nkx
    cdef int iz, iy, ix
    
    nkz, nky, nkx = len(kz2), len(ky2), len(kx2)
    cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)
    for iz in range(nkz):
        kz2i = kz2[iz]
        skz2i = skz2[iz]
        for iy in range(nky):
            ky2i = ky2[iy]
            skzy2i = skz2i*sky2[iy]
            for ix in range(nkx):
                k2_vals[iz,iy,ix] = (kz2i*skz2[iz]+ky2i*sky2[iy]+kx2[ix]*skx2[ix])
    # Avoid a divide by zero            
    k2_vals[0,0,0] = 1.

    return k2_vals
    
