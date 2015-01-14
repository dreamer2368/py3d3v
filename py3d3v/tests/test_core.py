
import numpy as np
import unittest
from nose.tools import set_trace
from core import *

norm = lambda x: np.max(np.abs(x))


class TestCalcE(unittest.TestCase):

    tol = 10e-6

    def test_calc_Ez_order(self):
        ny = nx = 10
        L = 1.

        k_vals = [9, 10]
        errors = []
        for k in k_vals:
            nz = 2**k
            dz, dy, dx = L/nz, L/ny, L/nx

            z_vals = np.sin(2*np.pi*np.linspace(0, L, nz+1)[:-1]/L)
            expected = -2*np.pi*np.cos(2*np.pi*np.linspace(0, L, nz+1)[:-1]/L)/L

            phi = np.zeros((nz, ny, nx))
            Ez_expected = np.zeros_like(phi)
            for i in range(nz):
                phi[i,:,:] = z_vals[i]
                Ez_expected[i,:,:] = expected[i]

            Ez = calc_Ez(phi, dz)
            errors.append(norm(Ez-Ez_expected))

        errors = np.log2(errors)
        order = -(errors[-1]-errors[-2])

        self.assertTrue(np.abs(order-2.)<self.tol)

    def test_calc_Ey_order(self):
        nz = nx = 10
        L = 1.

        k_vals = [9, 10]
        errors = []
        for k in k_vals:
            ny = 2**k
            dz, dy, dx = L/nz, L/ny, L/nx

            y_vals = np.sin(2*np.pi*np.linspace(0, L, ny+1)[:-1]/L)
            expected = -2*np.pi*np.cos(2*np.pi*np.linspace(0, L, ny+1)[:-1]/L)/L

            phi = np.zeros((nz, ny, nx))
            Ey_expected = np.zeros_like(phi)
            for i in range(ny):
                phi[:,i,:] = y_vals[i]
                Ey_expected[:,i,:] = expected[i]

            Ey = calc_Ey(phi, dy)
            errors.append(norm(Ey-Ey_expected))

        errors = np.log2(errors)
        order = -(errors[-1]-errors[-2])

        self.assertTrue(np.abs(order-2.)<self.tol)

    def test_calc_Ex_order(self):
        nz = ny = 10
        L = 1.

        k_vals = [9, 10]
        errors = []
        for k in k_vals:
            nx = 2**k
            dz, dy, dx = L/nz, L/ny, L/nx

            x_vals = np.sin(2*np.pi*np.linspace(0, L, nx+1)[:-1]/L)
            expected = -2*np.pi*np.cos(2*np.pi*np.linspace(0, L, nx+1)[:-1]/L)/L

            phi = np.zeros((nz, ny, nx))
            Ex_expected = np.zeros_like(phi)
            for i in range(nx):
                phi[:,:,i] = x_vals[i]
                Ex_expected[:,:,i] = expected[i]

            Ex = calc_Ex(phi, dx)
            errors.append(norm(Ex-Ex_expected))

        errors = np.log2(errors)
        order = -(errors[-1]-errors[-2])

        self.assertTrue(np.abs(order-2.)<self.tol)
