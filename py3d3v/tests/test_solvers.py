
import numpy as np
import unittest
from nose.tools import set_trace
from pic3d3v import *
from solvers import *
from landau import *

norm = lambda x: np.max(np.abs(x))


class TestShortRangeCalc(unittest.TestCase):

    tol = 10e-10

    def test_short_range_N_cells(self):
        """Compare single cell to many cell sr calcs

        Only checking default screen. This is only to check the short
        range logic, not the correctness of the screen functions.
        """
        np.random.seed(9999)
        screen = GaussianScreen
        N = 16**3
        Lz = Ly = Lx = np.pi
        xp = np.random.rand(N).astype(np.double)*Lx
        yp = np.random.rand(N).astype(np.double)*Ly
        zp = np.random.rand(N).astype(np.double)*Lz
        normalize(zp, Lz)
        normalize(yp, Ly)
        normalize(xp, Lx)
        q  = np.ones_like(zp, dtype=np.double)

        rmax = .2
        beta = 5.
        particles = Species(N, 1., 1., x0=xp, y0=yp, z0=zp)
        pic = PIC3DP3M([particles], (Lz,Ly,Lx), (32,32,32))
        pic.init_run(.1, beta=beta, rmax=rmax)
        pic.sort_by_cells()
        
        zp=pic.zp; yp=pic.yp; xp=pic.xp
        q=pic.q
        cell_span = pic.cell_span
        N_cells = pic.N_cells

        # Using only one cell
        Ezp1 = np.zeros(N, dtype=np.double)
        Eyp1 = np.zeros(N, dtype=np.double)
        Exp1 = np.zeros(N, dtype=np.double)

        screen.calc_E_short_range(Ezp1, zp, Lz, 
                                  Eyp1, yp, Ly,
                                  Exp1, xp, Lx, 
                                  q, 1, cell_span, 
                                  rmax, beta)

        # Using the default number of cells
        Ezp2 = np.zeros(N, dtype=np.double)
        Eyp2 = np.zeros(N, dtype=np.double)
        Exp2 = np.zeros(N, dtype=np.double)

        screen.calc_E_short_range(Ezp2, zp, Lz,
                                  Eyp2, yp, Ly,
                                  Exp2, xp, Lx,
                                  q, 
                                  N_cells, cell_span,
                                  rmax, beta)

        self.assertTrue(norm(Ezp1-Ezp2)<self.tol)
        self.assertTrue(norm(Eyp1-Eyp2)<self.tol)
        self.assertTrue(norm(Exp1-Exp2)<self.tol)
