

import numpy as np
import unittest
from nose.tools import set_trace
from pic3d3v import *
from tools import *


class TestSpecies(unittest.TestCase):

    def test_init_defaults(self):

        s = Species(3, 1., 2.)
        self.assertEqual(s.N, 3)
        self.assertEqual(s.q, 1.)
        self.assertEqual(s.m, 2.)
        expected = np.zeros(3, dtype=np.double)

        for i in [s.z0, s.y0, s.x0, s.vz0, s.vy0, s.vx0]:
            self.assertTrue((expected==i).all())

    def test_init_no_defaults(self):

        expected = np.ones(3, dtype=np.double)
        s = Species(3, 1., 2.,
                    z0=expected, y0=expected, x0=expected,
                    vz0=expected, vy0=expected, vx0=expected)

        for i in [s.z0, s.y0, s.x0, s.vz0, s.vy0, s.vx0]:
            self.assertTrue((expected==i).all())


class TestPICBase(unittest.TestCase):

    tol = 10e-10

    def setUp(self):
        s1 = Species(2, 2., 1.)
        s2 = Species(3, 4., 1.)
        self.p = PIC3DBase([s1, s2], (3., 4., 5.), (10, 20, 30))

    def test_unpack(self):
        ns1 = np.ones(2)
        s1 = Species(2, 1., 1.,
                     z0=ns1, y0=ns1, x0=ns1,
                     vz0=ns1, vy0=ns1, vx0=ns1)

        ns2 = np.ones(3)*2
        s2 = Species(3, 2., 2.,
                     z0=ns2, y0=ns2, x0=ns2,
                     vz0=ns2, vy0=ns2, vx0=ns2)

        ns3 = np.ones(2)*3
        s3 = Species(2, 3., 3.,
                     z0=ns3, y0=ns3, x0=ns3,
                     vz0=ns3, vy0=ns3, vx0=ns3)

        p = PIC3DBase([s1,s2,s3], (1., 1., 1.), (10, 10, 10))

        expected = np.array([1,1,2,2,2,3,3])

        for i in [p.zp, p.yp, p.xp, p.vz, p.vy, p.vx]:
            self.assertTrue((expected==i).all())

        self.assertTrue((p.q==np.array([1,1,2,2,2,3,3])).all())
        self.assertTrue((p.qm==np.array([1.,1.,1.,1.,1.,1.,1.])).all())

    def test_accel(self):
        p = self.p
        Ez = np.ones_like(p.q)
        Ey = np.ones_like(p.q)
        Ex = np.ones_like(p.q)
        dt = .5

        p.accel(Ez, Ey, Ex, dt)
        vz, vy, vx = p.vz, p.vy, p.vx
        expected = np.array([1., 1., 2., 2., 2.,])
        self.assertTrue((expected-vz<self.tol).all())
        self.assertTrue((expected-vy<self.tol).all())
        self.assertTrue((expected-vx<self.tol).all())

    def test_move(self):
        p = self.p
        Ez = np.ones_like(p.q)
        Ey = np.ones_like(p.q)
        Ex = np.ones_like(p.q)
        dt = .5
        p.accel(Ez, Ey, Ex, dt)
        p.move(dt)

        zp, yp, xp = p.zp, p.yp, p.xp
        expected = np.array([.5, .5, 1., 1., 1.,])
        self.assertTrue((expected-zp<self.tol).all())
        self.assertTrue((expected-yp<self.tol).all())
        self.assertTrue((expected-xp<self.tol).all())

    def test_rotate_none(self):
        """No rotation when B0==0
        """
        B0 = 0.
        vx0 = np.ones(2)
        vy0 = np.ones(2)*.5
        s1 = Species(2, 2., 1., vx0=vx0, vy0=vy0)
        p = PIC3DBase([s1], (3., 4., 5.), (10, 20, 30), B0=B0)

        p.rotate(1.)
        self.assertTrue((vx0==p.vx).all())
        self.assertTrue((vy0==p.vy).all())

    def test_rotate(self):
        """Check correct angle of rotation
        """
        B0 = .5
        vx0 = np.ones(2)
        vy0 = np.ones(2)*.5
        s1 = Species(2, 1., 1., vx0=vx0, vy0=vy0)
        p = PIC3DBase([s1], (3., 4., 5.), (10, 20, 30), B0=B0)
        p.rotate(np.pi/2.)

        # Expect a rotation of angle pi/2
        sq22 = np.sqrt(2.)/2.
        evx = sq22*(vx0+vy0)
        evy = sq22*(-vx0+vy0)
        
        self.assertTrue((evx-p.vx<self.tol).all())
        self.assertTrue((evy-p.vy<self.tol).all())

