import numpy as np
from linear import (LinearInterpolation,
                    BilinearInterpolation,
                    InterpolatePointsLinear,
                    InterpolatePointsBilinear)

class Interpolater(object):

    methods = ['linear']
    dtype = np.float64

    def __init__(self, values, dimensions, method):
        #self.values = np.array(values,dtype = self.dtype)
        self.values = values
        self.dimensions = dimensions
        self.n_dims = len(self.dimensions)
        
        if method in self.methods:
            self.method = method
        else:
            raise AttributeError("Invalid method")

    def scale_inputs(self, inputs):
        inputs = [np.array(k, dtype=self.dtype,ndmin=1)
                  for k in inputs]
        
        sizes = [len(k) for k in inputs]
        if len(np.unique([s for s in sizes if s!=1])) not in [0,1]:
            raise AttributeError("Inconsistent shapes in interpolate")

        max_size = max(sizes)
        if max_size!=1:
            for i in range(len(inputs)):
                if len(inputs[i])==1:
                    val = inputs[i]
                    inputs[i] = np.ndarray(max_size, dtype=self.dtype)
                    inputs[i][:] = val 
        
        scaled = []
        for i in range(len(inputs)):
            dim = self.dimensions[i]
            inp = inputs[i]
            scaled.append((inp-dim.start)/(dim.stop-dim.start)*(dim.steps-1))
        return [np.array(s,dtype=self.dtype)
                for s in scaled]

    def interpolate(self,*args):
        
        if len(args)!=self.n_dims:
            raise AttributeError("interpolate recieved %i args, expected %i" %
                                 (len(args),self.n_dims))

        x_vals, y_vals = self.scale_inputs(args)

        return InterpolatePointsBilinear(self.values, x_vals, y_vals)
        
    def __getitem__(self,item):
        x,y=item
        if isinstance(x,slice):
            x=np.linspace(x.start,x.stop,x.step)
        if isinstance(y,slice):
            y=np.linspace(y.start,y.stop,y.step)
        
        return self.interpolate(x,y)

