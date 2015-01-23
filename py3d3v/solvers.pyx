
import numpy as np
cimport numpy as np
import scipy as sp
from libc.math cimport floor, ceil, exp, erf, fabs, sqrt, cos, sin
from core import *

ctypedef np.float64_t DOUBLE

cdef extern from "par_solvers.hpp":

    void calc_E_short_range_par_gaussian(int N,
                                         double* Ezp, const double* zp, double Lz,
                                         double* Eyp, const double* yp, double Ly,
                                         double* Exp, const double* xp, double Lx,
                                         const double* q,
                                         long N_cells, const long* cell_span, 
                                         double rmax, double beta)

    void calc_E_short_range_par_s2(int N,
                                   double* Ezp, const double* zp, double Lz,
                                   double* Eyp, const double* yp, double Ly,
                                   double* Exp, const double* xp, double Lx,
                                   const double* q,
                                   long N_cells, const long* cell_span, 
                                   double rmax, double beta)

    void build_inf_lr_gaussian_optim_par(double* k2_vals,
                                         double* kz, int nkz, double dz,		
                                         double* ky, int nky, double dy,
                                         double* kx, int nkx, double dx,
                                         double beta, int m_max, int diff_order)

    void build_inf_lr_s2_optim_par(double* k2_vals,
                                   double* kz, int nkz, double dz,		
                                   double* ky, int nky, double dy,
                                   double* kx, int nkx, double dx,
                                   double beta, int m_max, int diff_order)


        
def get_k_vals(n, d):
    return (np.fft.fftfreq(n)*2*np.pi/d)


cpdef build_k2(int nz, double dz,
               int ny, double dy,
               int nx, double dx):
    """k2 for CIC weighting
    """
    cdef np.ndarray kz = get_k_vals(nz, dz)
    cdef np.ndarray ky = get_k_vals(ny, dy)
    cdef np.ndarray kx = get_k_vals(nx, dx)

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


cpdef build_k2_lr_s2(int nz, double dz,
                     int ny, double dy,
                     int nx, double dx,
                     double beta):
    """k2 for CIC weighting
    """
    cdef np.ndarray kz = get_k_vals(nz, dz)
    cdef np.ndarray ky = get_k_vals(ny, dy)
    cdef np.ndarray kx = get_k_vals(nx, dx)

    cdef np.ndarray kz2 = (kz)**2
    cdef np.ndarray ky2 = (ky)**2
    cdef np.ndarray kx2 = (kx)**2
    cdef double kz2i, ky2i

    cdef int nkz, nky, nkx
    cdef int iz, iy, ix, i, j, k

    cdef double c = 1.
    cdef double d = 12./(beta/2.)**4
    cdef double k2
    cdef double res, ex, kb2
    cdef double msz, msy, msx
    msz = 2*np.pi/dz
    msy = 2*np.pi/dy
    msx = 2*np.pi/dx
    cdef double kzi, kyi, kxi

    nkz, nky, nkx = len(kz2), len(ky2), len(kx2)
    cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)
    for iz in range(nkz):
        kz2i = kz2[iz]
        kzi  = kz[iz]
        for iy in range(nky):
            ky2i = ky2[iy]
            kyi  = ky[iy]
            for ix in range(nkx):
                k2 = (kz2i+ky2i+kx2[ix])
                kxi = kx[ix]
                if k2!=0:
                    res = 0
                    for i in range(0, 1):
                        for j in range(0, 1):
                            for k in range(0, 1):
                                ex = (kzi+i*msz)**2+(kyi+j*msy)**2+(kxi+k*msx)**2
                                kb2 = sqrt(ex)*beta/2.
                                res += d*(2.-2.*cos(kb2)-kb2*sin(kb2))/ex**3
                    k2_vals[iz,iy,ix] = c*res

    return k2_vals

cpdef build_k2_lr_gaussian(int nz, double dz,
                           int ny, double dy,
                           int nx, double dx,
                           double beta):
    """k2 for CIC weighting
    """
    cdef np.ndarray kz = get_k_vals(nz, dz)
    cdef np.ndarray ky = get_k_vals(ny, dy)
    cdef np.ndarray kx = get_k_vals(nx, dx)

    cdef np.ndarray kz2 = (kz)**2
    cdef np.ndarray ky2 = (ky)**2
    cdef np.ndarray kx2 = (kx)**2
    cdef double kz2i, ky2i

    cdef int nkz, nky, nkx
    cdef int iz, iy, ix, i, j, k

    cdef double c = 1.
    cdef double d = -1./beta**2/4.
    cdef double k2
    cdef double res, ex
    cdef double msz, msy, msx
    msz = 2*np.pi/dz
    msy = 2*np.pi/dy
    msx = 2*np.pi/dx
    cdef double kzi, kyi, kxi

    nkz, nky, nkx = len(kz2), len(ky2), len(kx2)
    cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)
    for iz in range(nkz):
        kz2i = kz2[iz]
        kzi  = kz[iz]
        for iy in range(nky):
            ky2i = ky2[iy]
            kyi  = ky[iy]
            for ix in range(nkx):
                k2 = (kz2i+ky2i+kx2[ix])
                kxi = kx[ix]
                if k2!=0:
                    res = 0
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            for k in range(-2, 3):
                                ex = (kzi+i*msz)**2+(kyi+j*msy)**2+(kxi+k*msx)**2
                                res += exp(d*ex)/ex
                    k2_vals[iz,iy,ix] = c*res

    return k2_vals


cpdef build_inf_lr_gaussian_optim(int nz, double dz,
                                  int ny, double dy,
                                  int nx, double dx,
                                  double beta, int m_max, int diff_order):

    cdef double[:] kz = get_k_vals(nz, dz)
    cdef double[:] ky = get_k_vals(ny, dy)
    cdef double[:] kx = get_k_vals(nx, dx)

    cdef int nkz = kz.shape[0]
    cdef int nky = ky.shape[0]
    cdef int nkx = kx.shape[0]
    cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)

    build_inf_lr_gaussian_optim_par(<double*>k2_vals.data,
                                    &kz[0], nkz, dz,		
                                    &ky[0], nky, dy,
                                    &kx[0], nkx, dx,
                                    beta, m_max, diff_order)


    return k2_vals
    
cpdef build_inf_lr_s2_optim(int nz, double dz,
                            int ny, double dy,
                            int nx, double dx,
                            double beta, int m_max, int diff_order):

    cdef double[:] kz = get_k_vals(nz, dz)
    cdef double[:] ky = get_k_vals(ny, dy)
    cdef double[:] kx = get_k_vals(nx, dx)

    cdef int nkz = kz.shape[0]
    cdef int nky = ky.shape[0]
    cdef int nkx = kx.shape[0]
    cdef np.ndarray k2_vals = np.zeros((nkz, nky, nkx), dtype=np.double)

    build_inf_lr_s2_optim_par(<double*>k2_vals.data,
                              &kz[0], nkz, dz,		
                              &ky[0], nky, dy,
                              &kx[0], nkx, dx,
                              beta, m_max, diff_order)


    return k2_vals
    

class Screen(object):
    pass

class GaussianScreen(Screen):

    @staticmethod
    def influence_function(int nz, double dz,
                           int ny, double dy,
                           int nx, double dx,
                           double beta, optimized=True, diff_order=None, m_max=2):

        if optimized:
            return build_inf_lr_gaussian_optim(nz, dz, ny, dy, nx, dx, beta,
                                               m_max, diff_order)
        else:
            return build_k2_lr_gaussian(nz, dz, ny, dy, nx, dx, beta)
        

    @staticmethod
    def calc_E_short_range(double[:] Ezp, double[:] zp, double Lz,
                           double[:] Eyp, double[:] yp, double Ly,
                           double[:] Exp, double[:] xp, double Lx,
                           double[:] q, long N_cells, long[:] cell_span, 
                           double rmax, double beta):
        
            cdef int N = len(zp)
            calc_E_short_range_par_gaussian(N,
                                            &Ezp[0], &zp[0], Lz,
                                            &Eyp[0], &yp[0], Ly,
                                            &Exp[0], &xp[0], Lx,
                                            &q[0],
                                            N_cells, &cell_span[0],
                                            rmax, beta)

class S2Screen(Screen):

    @staticmethod
    def influence_function(int nz, double dz,
                           int ny, double dy,
                           int nx, double dx,
                           double beta, optimized=True, diff_order=None, m_max=2):

        if optimized:
            return build_inf_lr_s2_optim(nz, dz, ny, dy, nx, dx,
                                         beta, m_max, diff_order)
        else:
            return build_k2_lr_s2(nz, dz, ny, dy, nx, dx, beta)

    @staticmethod
    def calc_E_short_range(double[:] Ezp, double[:] zp, double Lz,
                           double[:] Eyp, double[:] yp, double Ly,
                           double[:] Exp, double[:] xp, double Lx,
                           double[:] q, long N_cells, long[:] cell_span, 
                           double rmax, double beta):
        
            cdef int N = len(zp)
            calc_E_short_range_par_s2(N,
                                      &Ezp[0], &zp[0], Lz,
                                      &Eyp[0], &yp[0], Ly,
                                      &Exp[0], &xp[0], Lx,
                                      &q[0],
                                      N_cells, &cell_span[0],
                                      rmax, beta)


class Poisson3DFFT(object):
    
    def __init__(self, nz, dz, ny, dy, nx, dx):
        
        self.nz = nz; self.dz = dz
        self.ny = ny; self.dy = dy
        self.nx = nx; self.dx = dx
        self.k2 = build_k2(nz, dz, ny, dy, nx, dx)
        
    def solve(self, rho):
        
        rhok = np.fft.fftn(rho)
        phik = rhok/self.k2
        phik[0,0,0] = 0.
        phi = np.real(np.fft.ifftn(phik))
        Ez  = calc_Ez(phi, self.dz) 
        Ey  = calc_Ey(phi, self.dy)
        Ex  = calc_Ex(phi, self.dx)

        return (Ez, Ey, Ex)
    
    def __call__(self, rho):
        return self.solve(rho)

class Poisson3DFFTLR(object):
    
    def __init__(self, nz, dz, ny, dy, nx, dx, beta,
                 screen=GaussianScreen,
                 diff_type="fd"):
        
        self.nz = nz; self.dz = dz
        self.ny = ny; self.dy = dy
        self.nx = nx; self.dx = dx
        self.beta = beta

        if diff_type=="fd":
            self.diff_func = self.diff_fd
            self.diff_order = 1
        elif diff_type=="spectral":
            self.diff_func = self.diff_spectral
            self.diff_order = 0
            # negative i k (nik)
            ni = np.complex(0, -1.)
            self.nikz = get_k_vals(nz, dz).reshape((nz,1,1))*ni
            self.niky = get_k_vals(ny, dy).reshape((1,ny,1))*ni
            self.nikx = get_k_vals(nx, dx).reshape((1,1,nx))*ni
        else:
            raise ValueError("Invalid diff function: %s"%(diff_type,))

        if hasattr(screen, "__len__"):
            self.screen = screen[0]
            self.screen_options= screen[1]
        else:
            self.screen = screen
            self.screen_options= {}
        self.k2 = self.screen.influence_function(nz, dz, ny, dy,
                                                 nx, dx, beta, diff_order=self.diff_order,
                                                 **self.screen_options)
            

    def diff_fd(self, phik):

        phi = np.real(np.fft.ifftn(phik))

        Ez  = calc_Ez(phi, self.dz) 
        Ey  = calc_Ey(phi, self.dy)
        Ex  = calc_Ex(phi, self.dx)

        return Ez, Ey, Ex

    def diff_spectral(self, phik):

        grid = np.zeros_like(phik)

        grid[:] = phik*self.nikz
        Ez = np.real(np.fft.ifftn(grid))

        grid[:] = phik*self.niky
        Ey = np.real(np.fft.ifftn(grid))

        grid[:] = phik*self.nikx
        Ex = np.real(np.fft.ifftn(grid))

        return Ez, Ey, Ex
        
    def solve(self, rho):
        
        rhok = np.fft.fftn(rho)
        phik = rhok*self.k2
        phik[0,0,0] = 0.
        #phi = np.real(np.fft.ifftn(phik))
        
        return self.diff_func(phik)
    
    def __call__(self, rho):
        return self.solve(rho)
    
