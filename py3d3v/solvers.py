
import numpy as np
from _tools import build_k2

class Poisson3DFFT(object):
    
    def __init__(self, nz, dz, ny, dy, nx, dx):
        
        self.nz = nz; self.dz = dz
        self.ny = ny; self.dy = dy
        self.nx = nx; self.dx = dx
        self.k2 = build_k2(nz, dz, ny, dy, nx, dx)
        
    # def build_k2(self):
    #     # We really only want to do this once
    #     nz, ny, nx = self.nz, self.ny, self.nx
    #     dz, dy, dx = self.dz, self.dy, self.dx
        
    #     kz = np.fft.fftfreq(nz)*2*np.pi/dz
    #     ky = np.fft.fftfreq(ny)*2*np.pi/dy
    #     kx = np.fft.fftfreq(nx)*2*np.pi/dx

    #     nkz, nky, nkx = len(kz), len(ky), len(kx)
    #     # A matrix to normalize rho values
    #     k2_vals = np.zeros((nkz, nky, nkx))

    #     for iz in range(nkz):
    #         for iy in range(nky):
    #             for ix in range(nkx):
    #                 k2_vals[iz,iy,ix] = kz[iz]**2+ky[iy]**2+kx[ix]**2
    #     # Avoid a divide by zero            
    #     k2_vals[0,0,0] = 1.
        
    #     self.k2 = k2_vals
        
    def solve(self, rho):
        
        rhok = np.fft.fftn(rho)
        phik = rhok/self.k2
        phik[0,0,0] = 0.
        phi = np.fft.ifftn(phik)
        return np.real(phi)
    
    def __call__(self, rho):
        return self.solve(rho)

        
