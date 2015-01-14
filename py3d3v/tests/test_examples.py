"""Compare numerical results with analytic solutions
"""

import numpy as np
import unittest
from nose.tools import set_trace
from pic3d3v import *
from core import *
from interp import *
from landau import *

norm = lambda x: np.max(np.abs(x))


class TestPICBaseExamples(unittest.TestCase):
    """Numerical examples from PIC base class
    """

    tol = 10e-6

    def test_static_E_SHM(self):
        """SHM with E held fixed
        """
        nt = 500
        dt = .01
        t_vals = np.linspace(0, dt*nt, nt+1)
        
        L = 2*np.pi
        xc = L/2.  # Center of SHM
        x0 = xc + np.arange(5)*.1 # Init x positions

        nz = ny = 128
        nx = 128
        dz, dy, dx = L/nz, L/ny, L/nx
        Ez = Ey = Ex = np.zeros((nz, ny, nx))
        x_vals = np.linspace(0, L, nx+1)[:-1]
        x_diff = x_vals - xc
        for i in range(nx):
            Ex[:,:,i] = x_diff[i]

        s = Species(len(x0), -1., 1., x0=np.array([x0]))
        pic = PIC3DBase([s], (L, L, L), (nz, ny, nx))

        xa = np.zeros((nt+1, s.N)) # x values over time
        # init half step back
        Ezp = interp_cic(Ez, pic.zp, dz, pic.yp, dy, pic.xp, dx)
        Eyp = interp_cic(Ey, pic.zp, dz, pic.yp, dy, pic.xp, dx)
        Exp = interp_cic(Ex, pic.zp, dz, pic.yp, dy, pic.xp, dx)
        pic.accel(Ezp, Eyp, Exp, -dt/2.)
        xa[0,:] = pic.xp

        # main loop
        for i in range(1, nt+1):
            Ezp = interp_cic(Ez, pic.zp, dz, pic.yp, dy, pic.xp, dx)
            Eyp = interp_cic(Ey, pic.zp, dz, pic.yp, dy, pic.xp, dx)
            Exp = interp_cic(Ex, pic.zp, dz, pic.yp, dy, pic.xp, dx)

            pic.accel(Ezp, Eyp, Exp, dt)
            pic.move(dt)
            xa[i,:] = pic.xp

        # Compare with expected
        wp       = s.q/s.m
        delta_x  = xa-xc
        expected = np.outer(np.cos(wp*t_vals), delta_x[0,:])
        self.assertTrue(norm(delta_x-expected)<self.tol)

    def test_wc_freq(self):
        """Match expected with experimental cyclotron frequency
        """
        nt = 500
        dt = .1
        t_vals = np.linspace(0, dt*nt, nt+1)

        L   = 2*np.pi
        B0  = 1.
        z0  = np.array([1.])
        x0  = np.array([L/2.+1])
        y0  = np.array([L/2.])
        vy0 = np.ones(len(z0))

        nz = ny = 128
        nx = 128
        dz, dy, dx = L/nz, L/ny, L/nx
        Ezp = Eyp = Exp = np.zeros((len(z0),))

        s = Species(len(x0), -1., 1., x0=x0, y0=y0, z0=z0, vy0=vy0)
        pic = PIC3DBase([s], (L,L,L), (nz, ny, nx), B0=B0)

        vxa = np.zeros(nt+1) # x values over time
        vya = np.zeros(nt+1)
        vza = np.zeros(nt+1)
        # init half step back
        pic.rotate(-dt/2.)
        pic.accel(Ezp, Eyp, Exp, -dt/2.)
        vza[0] = pic.vz
        vya[0] = pic.vy
        vxa[0] = pic.vx

        # main loop
        for i in range(1, nt+1):

            pic.accel(Ezp, Eyp, Exp, dt/2.)
            pic.rotate(dt)
            pic.accel(Ezp, Eyp, Exp, dt/2.)
            pic.move(dt)
            vza[i] = pic.vz
            vya[i] = pic.vy
            vxa[i] = pic.vx

        wc = pic.wc
        evy = np.cos(wc*(t_vals-dt/2.))
        evx = np.sin(wc*(t_vals-dt/2.))

        self.assertTrue(norm(vxa-evx)<self.tol)
        self.assertTrue(norm(vya-evy)<self.tol)

class TestLangmuirDispersion(unittest.TestCase):

    def test_p3m_dispersion(self):

        ntc = 10
        nx0 = 64
        Nx0 = int(nx0*.5)
        beta = 5

        for mode in [1, 5, 10]:

            solver = (PIC3DP3M, {"beta":beta, "rmax":.5})
            l3d = dispersion_calc(ntc, nx0, Nx0, mode, solver=solver)
            self.assertTrue(np.abs(l3d.omega-l3d.wpe)<.1)
