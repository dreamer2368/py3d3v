import numpy as np
import unittest
from py3d3v.interpolate.interpolater import Interpolater

def almost_equal(a, b, tol=10**-14):
    a = np.array(a)
    b = np.array(b)
    return np.all(np.abs(a-b)<tol)

class TestCreate(unittest.TestCase):

    def test_can_create(self):
        interp = Interpolater([0],[0],'linear')

    def test_invalid_method(self):
        self.assertRaises(AttributeError,
                          Interpolater,
                          [0],[0],'not-a-method')
    
class TestInterpolate(unittest.TestCase):

    def setUp(self):
        self.x_dim = (0,1,10)
        self.y_dim = (2,3,20)
        self.dims = [self.x_dim,
                     self.y_dim]
        x = np.linspace(*self.x_dim)
        y = np.linspace(*self.y_dim)
        X,Y = np.meshgrid(x,y)
        vals = np.transpose(X+Y)
        self.interp = Interpolater(vals,
                                   self.dims,
                                   'linear')

    def expected(self, x, y):
        """Expected interpolation values
        """
        return np.sum([x,y])

    def test_scale_inputs(self):
        """Check left, mid, and right values
        """
        vals = [np.array([d[0],
                          .5*(d[0]+d[1]),
                          d[1]])
                for d in self.dims]
        expected = [np.array([0,
                              .5*(d[2]-1),
                              d[2]-1])
                    for d in self.dims]
        check = self.interp.scale_inputs(vals)
        equal = True
        for i in range(len(check)):
            if not np.all(check[i]==expected[i]):
                equal = False
        self.assertTrue(equal)

    def test_fails_with_wrong_n_args(self):

        self.assertRaises(AttributeError,
                          self.interp.interpolate,
                          [0],[0],[0])
        
    def test_all_single_elem(self):
        """The case where all arguments are not slices
        """
        x,y = .5, 2.25
        self.assertTrue(almost_equal(self.interp.interpolate(x,y),
                                     self.expected(x,y)))

    def test_all_multiple_elems(self):

        x_vals = np.linspace(.25,.75,10)
        y_vals = np.linspace(2,2.9,10)

        expected = [self.expected(x,y)
                    for x,y in zip(x_vals,y_vals)]

        self.assertTrue(almost_equal(self.interp.interpolate(x_vals,y_vals),
                                     expected))

    def test_fails_with_inconsistent_elems(self):
        self.assertRaises(AttributeError,
                          self.interp.interpolate,
                          [0,1],[2,2.5,3])

        self.assertRaises(AttributeError,
                          self.interp.interpolate,
                          [0,.5,1],[2,3])

    def test_mixed_elems(self):

        x_vals = np.linspace(.25,.75,10)
        y_vals = np.linspace(2,2.9,30)
        x, y = .5, 2

        expected = [self.expected(k,y)
                    for k in x_vals]
        pred = self.interp.interpolate(x_vals,y)
        self.assertTrue(len(pred)==len(expected))
        self.assertTrue(almost_equal(pred, expected))

        expected = [self.expected(x,k)
                    for k in y_vals]
        pred = self.interp.interpolate(x,y_vals)
        self.assertTrue(len(pred)==len(expected))
        self.assertTrue(almost_equal(pred, expected))

    @unittest.skip("Deal with this later")
    def test_getitem_fails_with_bad_slice(self):

        self.assertRaises(AttributeError,
                          self.interp.__getitem__,
                          (slice(1,2,None),0))

    @unittest.skip("Deal with this later")
    def test_getitem_fails_wrong_n_args(self):
        # When called with only one item item is
        # not iterable
        # When called with multiple items it is
        # a tuple.
        pass

    def test_getitem_mixed(self):
        # I know this is bad practice, but refactoring
        # the tests is a task for another day

        x_vals = np.linspace(.25,.75,10)
        y_vals = np.linspace(2,2.9,30)
        x, y = .5, 2

        expected = [self.expected(k,y)
                    for k in x_vals]
        pred = self.interp[x_vals,y]
        self.assertTrue(len(pred)==len(expected))
        self.assertTrue(almost_equal(pred, expected))

        expected = [self.expected(x,k)
                    for k in y_vals]
        pred = self.interp[x,y_vals]
        self.assertTrue(len(pred)==len(expected))
        self.assertTrue(almost_equal(pred, expected))

        expected = self.expected(x,y)
        pred = self.interp[x,y]
        self.assertTrue(almost_equal(pred, expected))
