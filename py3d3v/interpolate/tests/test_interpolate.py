import numpy as np
import unittest
from pydd.interpolate.linear import (LinearInterpolation,
                                     BilinearInterpolation,
                                     InterpolatePointsLinear,
                                     InterpolatePointsBilinear)

class TestLinearInterpolate(unittest.TestCase):

    def setUp(self):
        self.x1, self.x2 = 1.0, 2.0
        self.f1, self.f2 = 3.0, 4.0

    def test_left_point(self):
        self.assertEqual(self.f1,
                         LinearInterpolation(self.x1,
                                             self.x1,
                                             self.x2,
                                             self.f1,
                                             self.f2))

    def test_right_point(self):
        self.assertEqual(self.f2,
                         LinearInterpolation(self.x2,
                                             self.x1,
                                             self.x2,
                                             self.f1,
                                             self.f2))

    def test_mid_point(self):
        mid_point = (self.x1+self.x2)/2
        mid_value = (self.f1+self.f2)/2
        self.assertEqual(mid_value,
                         LinearInterpolation(mid_point,
                                             self.x1,
                                             self.x2,
                                             self.f1,
                                             self.f2))

class TestBilinearInterpolation(unittest.TestCase):

    def setUp(self):
        self.x1,self.x2    = 1.0,2.0
        self.y1,self.y2    = 3.0,4.0
        self.f11, self.f12 = 5.0,6.0
        self.f21, self.f22 = 7.0,8.0

    def test_11(self):
        self.assertEqual(self.f11,
                         BilinearInterpolation(
                             self.x1, self.y1,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_12(self):
        self.assertEqual(self.f12,
                         BilinearInterpolation(
                             self.x1, self.y2,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_21(self):
        self.assertEqual(self.f21,
                         BilinearInterpolation(
                             self.x2, self.y1,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_22(self):
        self.assertEqual(self.f22,
                         BilinearInterpolation(
                             self.x2, self.y2,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_mid_x_y1(self):
        mid_x = (self.x1+self.x2)/2.0
        mid_f = (self.f11+self.f21)/2.0
        self.assertEqual(mid_f,
                         BilinearInterpolation(
                             mid_x, self.y1,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_mid_x_y2(self):
        mid_x = (self.x1+self.x2)/2.0
        mid_f = (self.f12+self.f22)/2.0
        self.assertEqual(mid_f,
                         BilinearInterpolation(
                             mid_x, self.y2,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_mid_y_x1(self):
        mid_y = (self.y1+self.y2)/2.0
        mid_f = (self.f11+self.f12)/2.0
        self.assertEqual(mid_f,
                         BilinearInterpolation(
                             self.x1, mid_y,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_mid_y_x2(self):
        mid_y = (self.y1+self.y2)/2.0
        mid_f = (self.f21+self.f22)/2.0
        self.assertEqual(mid_f,
                         BilinearInterpolation(
                             self.x2, mid_y,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_mid(self):
        mid_x = (self.x1+self.x2)/2.0
        mid_y = (self.y1+self.y2)/2.0
        mid_f = (self.f21+self.f22)/2.0 + \
                (self.f11+self.f12)/2.0
        mid_f/=2.0
        self.assertEqual(mid_f,
                         BilinearInterpolation(
                             mid_x, mid_y,
                             self.x1, self.x2,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f21, self.f22))

    def test_equal_x(self):
        mid_y = (self.y1+self.y2)/2.0
        mid_f = (self.f11+self.f12)/2.0
        self.assertEqual(mid_f,
                         BilinearInterpolation(
                             self.x1, mid_y,
                             self.x1, self.x1,
                             self.y1, self.y2,
                             self.f11, self.f12,
                             self.f11, self.f12))

    def test_equal_y(self):
        mid_x = (self.x1+self.x2)/2.0
        mid_f = (self.f12+self.f22)/2.0
        self.assertEqual(mid_f,
                         BilinearInterpolation(
                             mid_x, self.y2,
                             self.x1, self.x2,
                             self.y2, self.y2,
                             self.f12, self.f12,
                             self.f22, self.f22))

def almost_equal(a, b, tol=10**-14):
    return np.all(np.abs(a-b)<tol)

class TestInterpolatePointsLinear(unittest.TestCase):

    def test_against_linear_function(self):

        dim = (-1,1,10)
        vals = np.linspace(*dim)
        x_vals = np.array([0,.25,1,1.5,5,9])
        slope = (dim[1]-dim[0])/float(dim[2]-1)

        actual = np.array([slope*x + dim[0] for x in x_vals])
        ipl = InterpolatePointsLinear(vals,x_vals)

        self.assertTrue(almost_equal(actual,ipl))


class TestInterpolatePointsBilinear(unittest.TestCase):

    def test_against_linear_function(self):
        
        x_dim = (-1,1,10)
        y_dim = (2,3,20)

        x_slope = (x_dim[1]-x_dim[0])/float(x_dim[2]-1)
        y_slope = (y_dim[1]-y_dim[0])/float(y_dim[2]-1)

        x_vals = np.linspace(*x_dim)
        y_vals = np.linspace(*y_dim)

        x_pred = np.array([0,.5,1,1.5,5,7,7.7,9,1,1.4,0,0,3,9])
        y_pred = np.array([0,.5,1,1.5,5,7,7.7,9,10,10.1,12,15,15.6,19])

        expected = np.array([x_slope*x + x_dim[0] + y_slope*y + y_dim[0] 
                             for x,y in zip(x_pred,y_pred)])

        X,Y = np.meshgrid(x_vals,y_vals)
        vals = np.transpose(X+Y)
        pred = InterpolatePointsBilinear(vals,x_pred,y_pred)

        self.assertTrue(almost_equal(expected,pred))

if __name__ == '__main__':
    unittest.main()
        
