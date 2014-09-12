
import numpy as np
from interp import *
import unittest
norm = lambda x: np.max(np.abs(x))

class TestInterp(unittest.TestCase):

    tol = 10e-10

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

        c = interp_cic(vals, z_vals/dz, y_vals/dy, x_vals/dx)

        self.assertTrue( norm(c-expected) < self.tol )


class TestWeight(unittest.TestCase):

    tol = 10e-10

    def test_interior_points(self):
        nz, ny, nx = 2, 2, 2
        grid = np.zeros((nz, ny, nx))
        L = 1.
        dz, dy, dx = L/nz, L/ny, L/nx

        x_vals = np.array([.25])
        y_vals = np.array([.25])
        z_vals = np.array([.25])
        q_vals = np.ones_like(z_vals)

        weight_cic(grid, z_vals/dz, y_vals/dy, x_vals/dx, q_vals)

        self.assertTrue( (grid.flatten()-0.125<self.tol).all() )
        self.assertTrue( np.sum(grid)-1.0 < self.tol )

    def test_exterior_points(self):
        nz, ny, nx = 2, 2, 2
        grid = np.zeros((nz, ny, nx))
        L = 1.
        dz, dy, dx = L/nz, L/ny, L/nx

        x_vals = np.array([.75])
        y_vals = np.array([.75])
        z_vals = np.array([.75])
        q_vals = np.ones_like(z_vals)

        weight_cic(grid, z_vals/dz, y_vals/dy, x_vals/dx, q_vals)

        self.assertTrue( (grid.flatten()-0.125 < self.tol).all() )
        self.assertTrue( np.sum(grid)-1.0 < self.tol )

    def test_on_grid_point(self):
        nz, ny, nx = 2, 2, 2
        grid = np.zeros((nz, ny, nx))
        L = 1.
        dz, dy, dx = L/nz, L/ny, L/nx

        x_vals = np.array([.5])
        y_vals = np.array([.5])
        z_vals = np.array([.5])
        q_vals = np.ones_like(z_vals)

        weight_cic(grid, z_vals/dz, y_vals/dy, x_vals/dx, q_vals)
        expected = np.zeros_like(grid)
        expected[1,1,1] = 1.

        self.assertTrue( (grid-expected < self.tol).all() )
        self.assertTrue( np.sum(grid)-1.0 < self.tol )

    def test_multiple_points(self):
        nz, ny, nx = 2, 2, 2
        grid = np.zeros((nz, ny, nx))
        L = 1.
        dz, dy, dx = L/nz, L/ny, L/nx

        x_vals = np.array([.1, .2])
        y_vals = np.array([.1, .2])
        z_vals = np.array([.1, .2])
        q_vals = np.ones_like(z_vals)

        weight_cic(grid, z_vals/dz, y_vals/dy, x_vals/dx, q_vals)
        expected = np.zeros_like(grid)
        expected[0,0,0] = .8**3
        for z in [0,1]:
            for y in [0,1]:
                for x in [0,1]:
                    s = z+y+x
                    expected[z,y,x] = .8**(3-s)*.2**s \
                                    + .6**(3-s)*.4**s

        self.assertTrue( norm(grid-expected) < self.tol )
        self.assertTrue( np.abs(np.sum(grid)-2.0) < self.tol )


