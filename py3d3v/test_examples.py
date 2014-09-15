"""Compare numerical results with analytic solutions
"""

import numpy as np
import unittest
from nose.tools import set_trace
from pic3d3v import *
from tools import *
from interp import *

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
        Ezp = interp_cic(Ez, pic.zp/dz, pic.yp/dy, pic.xp/dx)
        Eyp = interp_cic(Ey, pic.zp/dz, pic.yp/dy, pic.xp/dx)
        Exp = interp_cic(Ex, pic.zp/dz, pic.yp/dy, pic.xp/dx)
        pic.accel(Ezp, Eyp, Exp, -dt/2.)
        xa[0,:] = pic.xp

        # main loop
        for i in range(1, nt+1):
            Ezp = interp_cic(Ez, pic.zp/dz, pic.yp/dy, pic.xp/dx)
            Eyp = interp_cic(Ey, pic.zp/dz, pic.yp/dy, pic.xp/dx)
            Exp = interp_cic(Ex, pic.zp/dz, pic.yp/dy, pic.xp/dx)

            pic.accel(Ezp, Eyp, Exp, dt)
            pic.move(dt)
            xa[i,:] = pic.xp

        # Compare with expected
        wp       = s.q/s.m
        delta_x  = xa-xc
        expected = np.outer(np.cos(wp*t_vals), delta_x[0,:])
        self.assertTrue(norm(delta_x-expected)<self.tol)
