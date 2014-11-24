from pyevtk.hl import pointsToVTK
from pyevtk.vtk import VtkGroup
import numpy as np

g = VtkGroup("./group")
for i in range(1,nt+1):
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    pointsToVTK("sim"+str(i), x, y, z, {"V":(x,y,z)})
    g.addFile("sim"+str(i)+".vtu", sim_time=i*l3d.dt)
g.save()