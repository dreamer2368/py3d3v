import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil

ctypedef np.float64_t DOUBLE

cdef double _LinearInterpolation(double x,
                        double x1,
                        double x2,
                        double f1,
                        double f2):
    cdef double c, r
    if x1==x2 or f1==f2:
        return f1
    c=(x-x1)/(x2-x1)
    r=(1-c)*f1+c*f2
    return r

cdef double _BilinearInterpolation(double x, double y,
                                   double x1, double x2,
                                   double y1, double y2,
                                   double f11, double f12,
                                   double f21, double f22):
    cdef double r

    if x1==x2:
        r = _LinearInterpolation(y,y1,y2,f11,f12)
    elif y1==y2:
        r = _LinearInterpolation(x,x1,x2,f11,f21)
    else:
        r = f11*(x2-x)*(y2-y)+ \
            f21*(x-x1)*(y2-y)+ \
            f12*(x2-x)*(y-y1)+ \
            f22*(x-x1)*(y-y1)
        r=r/((x2-x1)*(y2-y1))
    return r

def LinearInterpolation(double x,
                        double x1,
                        double x2,
                        double f1,
                        double f2):
    return _LinearInterpolation(x,x1,x2,f1,f2)
              
def BilinearInterpolation(double x, double y,
                          double x1, double x2,
                          double y1, double y2,
                          double f11, double f12,
                          double f21, double f22):
    return _BilinearInterpolation(x,y,x1,x2,y1,y2,
                                  f11,f12,f21,f22)


def InterpolatePointsLinear(np.ndarray[DOUBLE,ndim=1] vals,
                            np.ndarray[DOUBLE,ndim=1] x_vals):

    """Interpolate at multiple points

    Assumes that the inputs are already scaled so that x_vals
    corresponds to the integer indexes of vals
    """
    
    # Lower and upper bounds
    cdef double lb, ub
    cdef unsigned int n_vals = x_vals.shape[0]
    
    cdef unsigned int i
    cdef double x
    cdef np.ndarray[DOUBLE, ndim=1] res = np.ndarray(n_vals)
    for i in range(n_vals):
        x = x_vals[i]
        lb = floor(x)
        ub = ceil(x) # Makes the logic easy
        if lb==ub:
            res[i] = vals[lb]
        else:
            res[i] = _LinearInterpolation(x,lb,ub,vals[lb],vals[ub])
            
    return res  

def InterpolatePointsBilinear(np.ndarray[DOUBLE,ndim=2] vals,
                              np.ndarray[DOUBLE,ndim=1] x_vals,
                              np.ndarray[DOUBLE,ndim=1] y_vals):
    
    # Lower and upper bounds
    cdef double lb1, ub1, lb2, ub2
    cdef unsigned int n_vals = len(x_vals)
    
    cdef unsigned int i
    cdef double x, y
    cdef np.ndarray[DOUBLE, ndim=1] res = np.ndarray(n_vals)
    
    for i in range(n_vals):
        x = x_vals[i]
        y = y_vals[i]
        lb1, ub1 = floor(x), ceil(x)
        lb2, ub2 = floor(y), ceil(y)
        res[i] = _BilinearInterpolation(x, y,
                                        lb1, ub1, lb2, ub2,
                                        vals[lb1,lb2],vals[lb1,ub2], 
                                        vals[ub1,lb2],vals[ub1,ub2])
            
    return res
