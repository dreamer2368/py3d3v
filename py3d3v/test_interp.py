
import numpy as np
from interp import *
import unittest
norm = lambda x: np.max(np.abs(x))

class TestInterp(unittest.TestCase):

    def test_interior_points(self):
        
        nz, ny, nx = 10, 20, 30
        vals = np.zeros((nz, ny, nx))
        L = 1.
        dz, dy, dx = L/nz, L/ny, L/nx

        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    vals[z, y, x] = 3.*z*dz + 2.*y*dy + 1.*x*dx

        x_vals = np.array([0., .25, .5, .75, .9])
        y_vals = np.array([.9, .75, .5, .25, 0.])
        z_vals = np.array([.1, .2, .3, .4, .5])
        expected = 3.*z_vals + 2.*y_vals + 1.*x_vals

        c = TriLinearInterp(vals, z_vals/dz, y_vals/dy, x_vals/dx)

        self.assertTrue( norm(c-expected)<10e-10 )
