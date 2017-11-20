import pynbody
import numpy as np
#from enthought.mayavi import mlab


#gadget - C
#z0 - F


def index3dTO1d(ng,i,j,k):
    return i*ng**2 +j*ng +k
    
def index1dTO3d(ng, i1d):
    k = (i1d)%ng
    j = (i1d -k)/ng%ng
    i = (i1d -k-j*ng)/ng**2
    return i,j,k


def make_corrected_coord(q,x,L):
    dx = x - q
    dx = np.where(dx >  L/2, dx-L, dx)
    dx = np.where(dx < -L/2, dx+L, dx)
    return q + dx
#-------------------------------------------------------------- 
#==============================================================================
def parameters(f):     
    print 'loadable keys', f.loadable_keys()  # ['pos', 'vel', 'iord', 'mass']
    print 'ID', np.max(f['iord'])
    print 'position shape', np.shape(f['pos'])
   
#========================================


#==============================================================================
ngr = nGr =  128   # 128#  256#    
L =  100.#   200# 
res = 400 #kpc
#dirIn = '/Volumes/Scratch/nesar/Streams/cmpc/'
#disIn = './'
#dirOut = '/Users/nesar/Desktop/Streams/Gadget2npy/'
#dirOut = './'
ngr = nGr =   128      # 128#  256#    
L =  100.#   200# 

dir2 = str(L)+'Mpc'
dir3 = str(ngr)

#==============================================================================
fileIn = '/Users/Yuyu/Halo/GADGET/snapshot32_051'

fsIC =  pynbody.load(fileIn)

pos = fsIC['pos']
vel = fsIC['vel']
mass = fsIC['mass']

x = pos[:,0]
y = pos[:,1]
z = pos[:,2]
vx = vel[:,0]
vy = vel[:,1]
vz = vel[:,2]

simul = np.dstack((x,y,z,vx,vy,vz,mass))[0]
np.save(fileIn+'.npy', simul)