
import numpy as np
cimport numpy as np
from libc.math cimport erfc, exp, sin, sqrt
import cython

@cython.boundscheck(False)
def ewald_real_sum(double[:] zp, double Lz, 
                   double[:] yp, double Ly, 
                   double[:] xp, double Lx, 
                   double[:] q, double alpha, int mmax):
    
    cdef int N = len(zp)
    cdef np.ndarray[double,ndim=1] Ezp = np.zeros_like(zp)
    cdef np.ndarray[double,ndim=1] Eyp = np.zeros_like(yp)
    cdef np.ndarray[double,ndim=1] Exp = np.zeros_like(xp)
    
    cdef double a = 2.*alpha/np.sqrt(np.pi)
    cdef double b = 1./(4*np.pi)
    cdef double alpha2 = alpha**2
    
    # Loop over force on i due to j
    cdef double dz, dy, dx
    cdef double dzm, dym, dxm
    cdef double r, r2, Er
    cdef double Ez, Ey, Ex
    cdef int i, j, mz, my, mx
    for i in range(N-1):
        for j in range(i+1,N):
            
            if i!=j:
                Ez = Ey = Ex = 0.
                dz = zp[i]-zp[j]
                dy = yp[i]-yp[j]
                dx = xp[i]-xp[j]
                for mz in range(-mmax, mmax+1):
                    for my in range(-mmax, mmax+1):
                        for mx in range(-mmax, mmax+1):
                            dzm = dz + mz*Lz
                            dym = dy + my*Ly
                            dxm = dx + mx*Lx

                            r2 = dzm**2+dym**2+dxm**2
                            r  = sqrt(r2)
                            
                            Er = (a*exp(-alpha2*r2)\
                                  +erfc(alpha*r)/r)/r
                            
                            Er = Er/r
                            Ez += Er*dzm
                            Ey += Er*dym
                            Ex += Er*dxm

                Ezp[i] += q[j]*Ez
                Eyp[i] += q[j]*Ey
                Exp[i] += q[j]*Ex

                Ezp[j] -= q[i]*Ez
                Eyp[j] -= q[i]*Ey
                Exp[j] -= q[i]*Ex

    return (b*Ezp, b*Eyp, b*Exp)

@cython.boundscheck(False)
def ewald_recip_sum(double[:] zp, double Lz, 
                    double[:] yp, double Ly, 
                    double[:] xp, double Lx, 
                    double[:] q, double alpha, int kmax):
    
    cdef int N = len(zp)
    cdef np.ndarray[double,ndim=1] Ezp = np.zeros_like(zp)
    cdef np.ndarray[double,ndim=1] Eyp = np.zeros_like(yp)
    cdef np.ndarray[double,ndim=1] Exp = np.zeros_like(xp)
    cdef double sz = 2*np.pi/Lz
    cdef double sy = 2*np.pi/Ly
    cdef double sx = 2*np.pi/Lx
    
    cdef double a = 1./(Lz*Ly*Lx)
    cdef double b = 1./(4*alpha**2)
    
    # Loop over force on i due to j
    cdef double dz, dy, dx
    cdef double r, r2
    cdef double Ez, Ey, Ex, E
    cdef double kz, ky, kx
    cdef double k2, kr
    cdef int i, j, kzi, kyi, kxi
    for i in range(N-1):
        for j in range(i+1,N):
            
            if i!=j:
                Ez = Ey = Ex = 0.
                dz = zp[i]-zp[j]
                dy = yp[i]-yp[j]
                dx = xp[i]-xp[j]
                for kzi in range(-kmax, kmax+1):
                    kz = kzi*sz
                    for kyi in range(-kmax, kmax+1):
                        ky = kyi*sy
                        for kxi in range(-kmax, kmax+1):
                            kx = kxi*sx
                            
                            if kzi!=0 or kyi!=0 or kxi!=0:
                                k2 = kz**2+ky**2+kx**2
                                kr = kz*dz+ky*dy+kx*dx

                                E = exp(-k2*b)*sin(kr)/k2

                                Ez += E*kz
                                Ey += E*ky
                                Ex += E*kx

                Ezp[i] += q[j]*Ez
                Eyp[i] += q[j]*Ey
                Exp[i] += q[j]*Ex
                
                Ezp[j] -= q[i]*Ez
                Eyp[j] -= q[i]*Ey
                Exp[j] -= q[i]*Ex

    return (a*Ezp, a*Eyp, a*Exp)

def ewald_sum(double[:] zp, double Lz, 
              double[:] yp, double Ly, 
              double[:] xp, double Lx, 
              double[:] q, double alpha, 
              int kmax, int mmax):
    
    Ezp1, Eyp1, Exp1 = ewald_real_sum(zp, Lz, yp, Ly, xp, 
                                      Lx, q, alpha, mmax)
    
    Ezp2, Eyp2, Exp2 = ewald_recip_sum(zp, Lz, yp, Ly, xp, Lx, 
                                       q, alpha, kmax)
    
    return (Ezp1+Ezp2, Eyp1+Eyp2, Exp1+Exp2)