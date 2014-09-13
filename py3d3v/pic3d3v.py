
import numpy as np

class Species(object):

    dtype = np.double

    def __init__(self, N, q, m,
                 z0=None,  y0=None,  x0=None,
                 vz0=None, vy0=None, vx0=None):

        self.N = N
        self.q = q
        self.m = m

        if z0=None:  self.z0  = np.zeros(N, dtype=self.dtype)
        if y0=None:  self.y0  = np.zeros(N, dtype=self.dtype)
        if x0=None:  self.x0  = np.zeros(N, dtype=self.dtype)
        if vz0=None: self.vz0 = np.zeros(N, dtype=self.dtype)
        if vy0=None: self.vy0 = np.zeros(N, dtype=self.dtype)
        if vx0=None: self.vx0 = np.zeros(N, dtype=self.dtype)


class PIC3DBase(object):

    build_solver = True
    B0 = 0.

    def __init__(self, species, dims, steps):

        self.dims  = dims
        self.Lz, self.Ly, self.Lx = self.dims
        self.steps = steps
        self.nz, self.ny, self.nx = self.steps

        self.species = species

        self.unpack()

    def unpack(self):

        species = self.species
        N = 0
        for s in species: N += s.N
        B0 = self.B0
        q, qm, wc, zp, yp, xp, vz, vy, vx = [np.zeros(N) for _ in range(9)]
        do_move = np.ndarray((N,), dtype=np.bool)
        do_move[:] = False
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
            do_move[count:count+s.N] = s.m>0
            count += s.N

        self.q,  self.qm, self.wc = q, qm, wc
        self.zp, self.yp, self.xp = zp, yp, xp
        self.vz, self.vy, self.vx = vz, vy, vx
        self.do_move = do_move

    def accel(self):
        pass

    def rotate(self):
        pass

    def move(self):
        pass
