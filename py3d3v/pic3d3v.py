
import numpy as np
from solvers import Poisson3DFFT
from interp import weight_cic, interp_cic
from tools import *

class Species(object):

    dtype = np.double

    def __init__(self, N, q, m,
                 z0=None,  y0=None,  x0=None,
                 vz0=None, vy0=None, vx0=None):

        self.N = N
        self.q = q
        self.m = m

        self.z0,  self.y0,  self.x0  =  z0,  y0,  x0
        self.vz0, self.vy0, self.vx0 = vz0, vy0, vx0
        if z0==None:
            self.z0  = np.zeros(N, dtype=self.dtype)
        if y0==None:
            self.y0  = np.zeros(N, dtype=self.dtype)
        if x0==None:
            self.x0  = np.zeros(N, dtype=self.dtype)
        if vz0==None:
            self.vz0 = np.zeros(N, dtype=self.dtype)
        if vy0==None:
            self.vy0 = np.zeros(N, dtype=self.dtype)
        if vx0==None:
            self.vx0 = np.zeros(N, dtype=self.dtype)


class PIC3DBase(object):

    def __init__(self, species, dims, steps, B0=0.):
        
        Lz, Ly, Lx = dims
        nz, ny, nx = steps
        dz, dy, dx = Lz/nz, Ly/ny, Lx/nx
        V = dz*dy*dx

        self.dims  = dims
        self.Lz, self.Ly, self.Lx = Lz, Ly, Lx
        self.steps = steps
        self.nz, self.ny, self.nx = nz, ny, nx
        self.dz, self.dy, self.dx = dz, dy, dx
        self.V = V
        self.species = species
        self.B0 = B0

        self.unpack()

    def unpack(self):

        species = self.species
        N = 0
        for s in species: N += s.N
        B0 = self.B0
        q, qm, wc, zp, yp, xp, vz, vy, vx = [np.zeros(N) for _ in range(9)]
        #do_move = np.ndarray((N,), dtype=np.bool)
        #do_move[:] = False
        count = 0 # Trailing count
        for s in species:
            q[count:count+s.N]  = s.q
            qm[count:count+s.N] = s.q/s.m
            wc[count:count+s.N] = (s.q/s.m)*B0
            zp[count:count+s.N] = s.z0
            yp[count:count+s.N] = s.y0
            xp[count:count+s.N] = s.x0
            vz[count:count+s.N] = s.vz0
            vy[count:count+s.N] = s.vy0
            vx[count:count+s.N] = s.vx0
            #do_move[count:count+s.N] = s.m>0
            count += s.N
        #do_move = do_move.astype(np.int32)

        q = np.ascontiguousarray(q)
        qm = np.ascontiguousarray(qm)
        wc = np.ascontiguousarray(wc)
        zp = np.ascontiguousarray(zp)
        yp = np.ascontiguousarray(yp)
        xp = np.ascontiguousarray(xp)
        vz = np.ascontiguousarray(vz)
        vy = np.ascontiguousarray(vy)
        vx = np.ascontiguousarray(vx)        

        self.q,  self.qm, self.wc = q, qm, wc
        self.zp, self.yp, self.xp = zp, yp, xp
        self.vz, self.vy, self.vx = vz, vy, vx
        #self.do_move = do_move

    def accel(self, Ez, Ey, Ex, dt):
        # qm = self.qm
        # self.vz[:] = self.vz + qm*Ez*dt
        # self.vy[:] = self.vy + qm*Ey*dt
        # self.vx[:] = self.vx + qm*Ex*dt

        accel(dt, self.qm,
              Ez, self.vz,
              Ey, self.vy,
              Ex, self.vx)

    def rotate(self, dt):
        """Still assuming the B is uniform in the z direction
        """
        c = np.cos(self.wc*dt)
        s = np.sin(self.wc*dt)
        vx_new =  c*self.vx + s*self.vy
        vy_new = -s*self.vx + c*self.vy

        self.vx[:] = vx_new
        self.vy[:] = vy_new

    def move(self, dt):
        """Move in place
        """
        #do_move    = self.do_move
        zp, yp, xp = self.zp, self.yp, self.xp
        vz, vy, vx = self.vz, self.vy, self.vx
        Lz, Ly, Lx = self.dims

        # zp = np.ascontiguousarray(zp)
        # yp = np.ascontiguousarray(yp)
        # xp = np.ascontiguousarray(xp)
        # vz = np.ascontiguousarray(vz)
        # vy = np.ascontiguousarray(vy)
        # vx = np.ascontiguousarray(vx)
        move(dt,
             zp, vz, Lz,
             yp, vy, Ly,
             xp, vx, Lx)


class PIC3DPM(PIC3DBase):

    def calc_E_at_points(self):
        zp, yp, xp = self.zp, self.yp, self.xp
        dz, dy, dx = self.dz, self.dy, self.dx
        nz, ny, nx = self.nz, self.ny, self.nx
        zps, yps, xps = zp/dz, yp/dy, xp/dx
        grid = self.grid

        # Calculate phi
        weight_cic(grid, zps, yps, xps, self.q)
        grid[:] = grid*(1./self.V)
        grid[:] = self.solver.solve(grid)

        # Calculate E fields at points
        Ez  = calc_Ez(grid, dz) 
        Ezp = interp_cic(Ez, zp, dz, yp, dy, xp, dx)
        Ey  = calc_Ey(grid, dy)
        Eyp = interp_cic(Ey, zp, dz, yp, dy, xp, dx)
        Ex  = calc_Ex(grid, dx)
        Exp = interp_cic(Ex, zp, dz, yp, dy, xp, dx)
        self.Ezp = Ezp
        self.Eyp = Eyp
        self.Exp = Exp
        return (Ezp, Eyp, Exp)

    def init_run(self, dt, unpack=False):
        if unpack:
            self.unpack()
        self.solver = Poisson3DFFT(self.nz, self.dz,
                                   self.ny, self.dy,
                                   self.nx, self.dx)
        self.grid = np.zeros((self.nz, self.ny, self.nx))

        Ezp, Eyp, Exp = self.calc_E_at_points()
        self.accel(Ezp, Eyp, Exp, -dt/2.)
    
    def single_step(self, dt):
        Ezp, Eyp, Exp = self.calc_E_at_points()
        self.accel(Ezp, Eyp, Exp, dt)
        self.move(dt)

