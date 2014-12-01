
import numpy as np
from tools import *
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

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
        phi = np.fft.ifftn(phik)
        return np.real(phi)
    
    def __call__(self, rho):
        return self.solve(rho)


class Poisson3DFFTLR(object):
    
    def __init__(self, nz, dz, ny, dy, nx, dx, beta, screen="gaussian"):
        
        self.nz = nz; self.dz = dz
        self.ny = ny; self.dy = dy
        self.nx = nx; self.dx = dx
        self.beta = beta
        self.screen = screen
        if screen=="gaussian":
            self.k2 = build_k2_lr_gaussian(nz, dz, ny, dy, nx, dx, beta)
        elif screen=="S2":
            self.k2 = build_k2_lr_s2(nz, dz, ny, dy, nx, dx, beta)
        
    def solve(self, rho):
        
        rhok = np.fft.fftn(rho)
        phik = rhok*self.k2
        phik[0,0,0] = 0.
        phi = np.fft.ifftn(phik)
        return np.real(phi)
    
    def __call__(self, rho):
        return self.solve(rho)
        

def three_d_poisson(nz, dz, ny, dy, nx, dx, as_csr=True):

    kx = 1./(dx**2)
    ky = 1./(dy**2)
    kz = 1./(dz**2)

    N = nx*ny*nz
    A = sp.sparse.lil_matrix((N, N), dtype=np.double)
    
    for k in xrange(nz):
        for i in xrange(ny):
            for j in xrange(nx):
                
                # row/col for point
                m = j+i*nx+k*nx*ny 
                
                # x-direction
                A[m, m] -= -2*kx
                if m+1<N:                
                    A[m, m+1] -= 1*kx
                if m-1>=0:                
                    A[m, m-1] -= 1*kx
                    
                # y-direction
                A[m, m] -= -2*ky
                if m+nx<N:                
                    A[m, m+nx] -= 1*ky
                if m-nx>=0:                
                    A[m, m-nx] -= 1*ky
                    
                # z-direction
                A[m, m] -= -2*kz
                if m+nx*ny<N:
                    A[m, m+nx*ny] -= 1*kz
                if m-nx*ny>=0:
                    A[m, m-nx*ny] -= 1*kz
                    
    if as_csr:
        return A.tocsr()
    else:
        return A


class Poisson3DFD(object):
    
    def __init__(self, nz, dz, ny, dy, nx, dx):
        
        self.nz = nz; self.dz = dz
        self.ny = ny; self.dy = dy
        self.nx = nx; self.dx = dx
        self.A = three_d_poisson(nz, dz, ny, dy, nx, dx)
        
    def solve(self, rho):
        
        return sp.sparse.linalg.spsolve(self.A, rho)
    
    def __call__(self, rho):
        return self.solve(rho)
        
