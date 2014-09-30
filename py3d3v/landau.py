
import numpy as np
from tools import *
from pic3d3v import *
from interp import *
from solvers import *

class Landau3D(object):
    
    def __init__(self, 
                 part_dims = (128, 128, 128),
                 grid_dims = (64, 64, 64),
                 cube_dims = (2*np.pi, 2*np.pi, 2*np.pi),
                 mode = 1, A = 0.01, wp = 1., B0 = 0, 
                 nt = 50, dt = .1, save_xa=False, save_rr=False):
        
        self.Nz, self.Ny, self.Nx = part_dims
        self.N = self.Nz*self.Ny*self.Nx
        self.nz, self.ny, self.nx = grid_dims
        self.Lz, self.Ly, self.Lx = cube_dims
        self.mode = mode
        self.A   = A
        self.wp  = wp
        self.wp2 = wp**2
        self.B0  = B0
        self.nt  = nt
        self.dt  = dt
        self.save_xa = save_xa
        self.save_rr = save_rr
        
    def init_run(self):
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        nz, ny, nx = self.nz, self.ny, self.nx
        Lz, Ly, Lx = self.Lz, self.Ly, self.Lx
        mode, N = self.mode, self.N
        A, wp   = self.A, self.wp
        nt, dt  = self.nt, self.dt
        
        #t_vals = np.linspace(0, dt*nt, nt+1)
        dz, dy, dx = Lz/nz, Ly/ny, Lx/nx
        V = dz*dy*dx 
        n0       = Nx/Lx
        k        = mode*2*np.pi/Lx
        x0       = np.linspace(0., Lx, Nx+1)[:-1]
        x1       = A*np.cos(x0*k)
        init_pos = x0 + x1
        x0i      = np.linspace(0., Lx, Nx+1)[:-1]
        normalize(init_pos, Lx)
        y0  = np.linspace(0., Ly, Ny+1)[:-1]
        z0  = np.linspace(0., Lz, Nz+1)[:-1]
        z0a = np.zeros(N)
        y0a = np.zeros(N)
        x0a = np.zeros(N)
        ind = 0
        for iz in range(Nz):
            for iy in range(Ny):
                for ix in range(Nx):
                    z0a[ind] = z0[iz]
                    y0a[ind] = y0[iy]
                    x0a[ind] = init_pos[ix]
                    ind += 1

        q0 = wp**2/n0
        m0 = q0
        q = q0*Lz*Ly/(Ny*Nz)
        m = q

        electron = Species(N, -q, m,   x0=x0a, z0=z0a, y0=y0a)
        species = [electron]

        pic = PIC3DPM(species, (Lz, Ly, Lx), (nz, ny, nx))
        
        pic.init_run(dt)

        self.pic = pic
        self.dz, self.dy, self.dx = dz, dy, dx
        self.k, self.V = k, V
        if self.save_xa:
            xa = np.zeros((nt+1, N))
            xa[0,:]  = pic.xp[:]
            self.xa = xa
        if self.save_rr:
            self.rr    = np.zeros(nt+1)
            self.grho  = np.zeros(nx)
            self.rr[0] = self.calc_rr()

    def calc_rr(self):
        mode, dx, q = self.mode, self.dx, self.pic.q
        grho, rr, xp = self.grho, self.rr, self.pic.xp
        weight_cic_1d(grho, xp, dx, q)
        rhok  = np.fft.rfft(grho)
        return np.real(rhok[mode]*rhok[mode].conj())

    def run(self):
        nt, dt  = self.nt, self.dt
        pic = self.pic
        for i in range(1, nt+1):
            pic.single_step(dt)
            if self.save_xa:
                self.xa[i,:] = pic.xp[:]
            if self.save_rr:
                self.rr[i] = self.calc_rr()
