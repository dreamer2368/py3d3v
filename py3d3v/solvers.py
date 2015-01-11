
import numpy as np
from core import *

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
        else:
            assert False, "Invalid Screen %s"%(screen,)
        
    def solve(self, rho):
        
        rhok = np.fft.fftn(rho)
        phik = rhok*self.k2
        phik[0,0,0] = 0.
        phi = np.fft.ifftn(phik)
        return np.real(phi)
    
    def __call__(self, rho):
        return self.solve(rho)
