import random
from random import *
import math
from math import *
import numpy as np
from numpy import *
import time

points = 100

halos=[0,100.,150.]

x=[]
y=[]
z=[]
id=[]
for i in arange(0,points,1):
   x.append(halos[0]+ np.random.rand())
   y.append(halos[0]+np.random.rand())
   z.append(halos[0]+np.random.rand())
   id.append(i)

for i in arange(points,points*2,1):
   x.append(halos[1]+np.random.rand())
   y.append(halos[1]+np.random.rand())
   z.append(halos[1]+np.random.rand())
   id.append(i)

for i in arange(points*2,points*3,1):
   x.append(halos[2]+np.random.rand())
   y.append(halos[2]+np.random.rand())
   z.append(halos[2]+np.random.rand())
   id.append(i)





















x=array(x)
y=array(y)
z=array(z)
id=array(id)

t0 = time.time()                         

id_grp=[]
groups=zeros((len(x),1)).tolist()
particles=id
b=1 # linking length
while len(particles)>0:
  index = particles[0]
# remove the particle from the particles list
  #particles.remove(index)
  np.delete(particles, index)
  groups[index]=[index]
  print "#N ", index
  dx=x-x[index]
  dy=y-y[index]
  dz=z-z[index]
  dr=sqrt(dx**2.+dy**2.+dz**2.)
  id_to_look = np.where(dr<b)[0].tolist()
  id_to_look.remove(index)
  nlist = id_to_look
  # remove all the neighbors from the particles list
  for i in nlist:
        if (i in particles):
           np.delete(particles, i)
  print "--> neighbors", nlist
  groups[index]=groups[index]+nlist
  new_nlist = nlist
  while len(new_nlist)>0:
          index_n = new_nlist[0]
          np.delete(new_nlist, index_n)
          print "----> neigh", index_n
          dx=x-x[index_n]
          dy=y-y[index_n]
          dz=z-z[index_n]
          dr=sqrt(dx**2.+dy**2.+dz**2.)
          id_to_look = where(dr<b)[0].tolist()
          id_to_look = list(set(id_to_look) & set(particles))
          nlist = id_to_look
          if (len(nlist)==0):
             print "No new neighbors found"
          else:
             groups[index]=groups[index]+nlist
             new_nlist=new_nlist+nlist
             print "------> neigh-neigh", new_nlist
             for k in nlist:
               np.delete(particles, k)
