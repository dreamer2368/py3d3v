
import numpy as np
from _tools import *

        
def calc_Ez(phi, dz):
    
    E           = np.zeros_like(phi)
    E[1:-1,:,:] = -(phi[2:,:,:]-phi[:-2,:,:])
    E[0,:,:]    = -(phi[1,:,:]-phi[-1,:,:])
    E[-1,:,:]   = -(phi[0,:,:]-phi[-2,:,:])
    
    return E/(2*dz)

def calc_Ey(phi, dy):
    
    E           = np.zeros_like(phi)
    E[:,1:-1,:] = -(phi[:,2:,:]-phi[:,:-2,:])
    E[:,0,:]    = -(phi[:,1,:]-phi[:,-1,:])
    E[:,-1,:]   = -(phi[:,0,:]-phi[:,-2,:])
    
    return E/(2*dy)

def calc_Ex(phi, dx):
    
    E           = np.zeros_like(phi)
    E[:,:,1:-1] = -(phi[:,:,2:]-phi[:,:,:-2])
    E[:,:,0]    = -(phi[:,:,1]-phi[:,:,-1])
    E[:,:,-1]   = -(phi[:,:,0]-phi[:,:,-2])
    
    return E/(2*dx)
    
