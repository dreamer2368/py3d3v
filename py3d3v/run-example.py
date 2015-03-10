import numpy as np
from landau import *

rmax = .5
L    = 2*np.pi
ntc  = 10
nx0  = 64
Nx0  = int(nx0*.5)
beta = 4
mode = 1

# Change these to test different code paths
shape  = 2         # From 2-5
method = PIC3DP3M # PIC3DP3M or PIC3DPM
    
solver = (method, {"beta":beta, "rmax":rmax,
                   "solver_opts":{"diff_type":"spectral"},
                   "particle_shape":shape})

l3d = dispersion_calc(ntc, nx0, Nx0, mode, L=L, solver=solver)    

print l3d.omega
