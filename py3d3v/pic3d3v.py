
import numpy as np
import scipy as np
from solvers import *
from interp import weight_cic, interp_cic
from core import *

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
        if z0 is None:
            self.z0  = np.zeros(N, dtype=self.dtype)
        if y0 is None:
            self.y0  = np.zeros(N, dtype=self.dtype)
        if x0 is None:
            self.x0  = np.zeros(N, dtype=self.dtype)
        if vz0 is None:
            self.vz0 = np.zeros(N, dtype=self.dtype)
        if vy0 is None:
            self.vy0 = np.zeros(N, dtype=self.dtype)
        if vx0 is None:
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
        self.N = N
        B0 = self.B0
        q, qm, wc, zp, yp, xp, vz, vy, vx = [np.zeros(N) for _ in range(9)]
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
            count += s.N

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

    def accel(self, Ez, Ey, Ex, dt):

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

    def weight(self, grid, zp, dz, yp, dy,
               xp, dx, q, rho0=0.):

        weight_cic(grid, zp, dz, yp, dy, xp, dx, q)

    def interp(self, E, zp, dz, yp, dy, xp, dx):

        return interp_cic(E, zp, dz, yp, dy, xp, dx)

    
    def calc_E_at_points(self):
        zp, yp, xp = self.zp, self.yp, self.xp
        dz, dy, dx = self.dz, self.dy, self.dx
        nz, ny, nx = self.nz, self.ny, self.nx
        grid = self.grid

        # Calculate phi
        self.weight(grid, zp, dz, yp, dy, xp, dx, self.q)
        grid[:] = grid*(1./self.V)
        Ez, Ey, Ex = self.solver.solve(grid)

        # Calculate E fields at points
        Ezp = self.interp(Ez, zp, dz, yp, dy, xp, dx)
        Eyp = self.interp(Ey, zp, dz, yp, dy, xp, dx)
        Exp = self.interp(Ex, zp, dz, yp, dy, xp, dx)
        self.Ezp = Ezp
        self.Eyp = Eyp
        self.Exp = Exp
        return (Ezp, Eyp, Exp)

    def init_run(self, dt, unpack=False, particle_shape=2):
        if unpack:
            self.unpack()
        self.particle_shape = particle_shape
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


class PIC3DP3M(PIC3DPM):

    def calc_E_at_points(self):

        # Make sure particles are ordered corectly for
        # short range force calculation
        if self.N_cells>1:
            self.sort_by_cells()

        zp, yp, xp = self.zp, self.yp, self.xp
        dz, dy, dx = self.dz, self.dy, self.dx
        nz, ny, nx = self.nz, self.ny, self.nx
        Lz, Ly, Lx = self.Lz, self.Ly, self.Lx
        q = self.q
        grid = self.grid

        # Calculate long range forces

        ## Calculate phi
        self.weight(grid, zp, dz, yp, dy, xp, dx, self.q)
        grid[:] = grid*(1./self.V)
        Ez, Ey, Ex = self.solver.solve(grid)

        ## Calculate E fields at points
        Ezp = self.interp(Ez, zp, dz, yp, dy, xp, dx)
        Eyp = self.interp(Ey, zp, dz, yp, dy, xp, dx)
        Exp = self.interp(Ex, zp, dz, yp, dy, xp, dx)

        # Calculate short range forces
        self.screen.calc_E_short_range(Ezp, zp, Lz,
                                       Eyp, yp, Ly,
                                       Exp, xp, Lx, q,
                                       self.N_cells, self.cell_span,
                                       self.rmax, self.beta)

        # Return results
        self.Ezp = Ezp
        self.Eyp = Eyp
        self.Exp = Exp
        return (Ezp, Eyp, Exp)

    def init_run(self, dt, beta=10, rmax=.2,
                 screen=GaussianScreen,
                 N_cells=None, unpack=False,
                 solver_opts={},
                 particle_shape=2):

        if unpack:
            self.unpack()

        solver_opts["screen"] = screen
        solver_opts["beta"]   = beta
        solver_opts["particle_shape"] = particle_shape
        self.solver_opts = solver_opts
        self.solver = Poisson3DFFTLR(self.nz, self.dz,
                                     self.ny, self.dy,
                                     self.nx, self.dx,
                                     **self.solver_opts)
        self.screen = self.solver.screen

        # Choose the largest number of cells possible
        if N_cells is None:
            Lmin = np.min([self.Lz, self.Ly, self.Lx])
            N_cells = int(np.floor(Lmin/rmax))

        self.beta = beta
        self.rmax = rmax
        self.N_cells = N_cells
        self.cell_vals = np.arange(N_cells**3,  dtype=np.int)
        self.cell_span = np.zeros(N_cells**3+1, dtype=np.int)
        self.grid = np.zeros((self.nz, self.ny, self.nx))
        self.cell = np.zeros_like(self.zp, dtype=np.double)

        Ezp, Eyp, Exp = self.calc_E_at_points()
        self.accel(Ezp, Eyp, Exp, -dt/2.)

    def sort_by_cells(self):
        zp, yp, xp = self.zp, self.yp, self.xp
        vz, vy, vx = self.vz, self.vy, self.vx
        Lz, Ly, Lx = self.Lz, self.Ly, self.Lx
        cell       = self.cell
        cell_span  = self.cell_span
        N_cells    = self.N_cells

        Cz = Lz/N_cells
        Cy = Ly/N_cells
        Cx = Lx/N_cells

        zp_cell = np.trunc(zp/Cz)
        yp_cell = np.trunc(yp/Cy)
        xp_cell = np.trunc(xp/Cx)

        cell[:] = xp_cell+yp_cell*N_cells+zp_cell*N_cells**2

        s = cell.argsort()
        zp[:] = zp[s]
        yp[:] = yp[s]
        xp[:] = xp[s]
        vz[:] = vz[s]
        vy[:] = vy[s]
        vx[:] = vx[s]
        cell[:] = cell[s]
        cell_span[:-1] = np.searchsorted(cell, self.cell_vals)
        cell_span[-1]  = self.N

