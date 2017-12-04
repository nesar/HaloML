"""
./FoF_Special . snapshot 005

./FoF_Special . snap100Mpc128_z0 000



"""
import numpy as np
#from numpy import *

#=============================================
max_axis_lim = 50.
min_axis_lim = -50.
shift_position_constant = 100.

bins=0.2   
tbins=1.5
n_bin = 30

intervals = np.logspace(np.log10(bins), np.log10(tbins), num = n_bin)
radius = intervals[1:]

d_critical = 1.3598e11
m_particle = 3.65235543e10

tolerance =1.e11
tolerance_r = 1.e-5

#intervals = np.array([nn*bins for nn in range(1,int(tbins/bins)+1)])
#radius = intervals[1:]
#=============================================

def read_groups_catalogue(filename):
  """
  Read the "fof_special_catalogue" files and return 4 arrays:
  
  GroupLen	: the size of each group
  GroupOffset	: the offset of the first particle in each group
                  as it will be found in the file fof_special_particles
  GroupMass	: the mass of each group (in code unit)
  GroupCM	: the center of mass of each group
  """


  f = open(filename,'r')

  Ngroups = np.fromstring(f.read(4),np.int32)[0]

  GroupLen    = np.fromstring(f.read(4*Ngroups),np.int32)
  GroupOffset = np.fromstring(f.read(4*Ngroups),np.int32)
  GroupMass   = np.fromstring(f.read(4*Ngroups),np.float32)

  GroupCM     = np.fromstring(f.read(3*4*Ngroups),np.float32)
  GroupCM.shape  = (Ngroups,3)


  GroupNspecies = np.fromstring(f.read(3*4*Ngroups),np.int32)
  GroupNspecies.shape  = (Ngroups,3)

  GroupMspecies = np.fromstring(f.read(3*4*Ngroups),np.float32)
  GroupMspecies.shape  = (Ngroups,3)

  GroupSfr   = np.fromstring(f.read(4*Ngroups),np.float32)

  GroupMetallicities = np.fromstring(f.read(2*4*Ngroups),np.float32)
  GroupMetallicities.shape  = (Ngroups,2)

  Mcold = np.fromstring(f.read(4*Ngroups),np.float32) 

  SigmaStars= np.fromstring(f.read(4*Ngroups),np.float32) 
  SigmaDM= np.fromstring(f.read(4*Ngroups),np.float32) 

  f.close()

  return GroupLen,GroupOffset,GroupMass,GroupCM, GroupNspecies, GroupSfr
  
  
  
  
def read_groups_particles(filename):
  """
  Read the "fof_special_particles" files and return
  an array of the positions of each particles belonging
  to a group.
  """
  
  f = open(filename,'r')

  Ntot = np.fromstring(f.read(4),np.int32)[0]
  Pos	  = np.fromstring(f.read(3*4*Ntot),np.float32)
  Pos.shape  = (Ntot,3)
  f.close()
  
  return Pos, Ntot
  
  
def read_groups_indexlist(filename):
  """
  Read the "fof_special_particles" files and return
  an array of the positions of each particles belonging
  to a group.
  """
  
  f = open(filename,'r')
  
  Ntot = np.fromstring(f.read(4),np.int32)[0]
  idx	  = np.fromstring(f.read(3*4*Ntot),np.float32)
  
  f.close()
  
  return Ntot, idx

def distance_calc(gals_pos,sph_pos, axis):
    dxyzd = gals_pos[None, :, axis] - sph_pos[None, axis]
    dxyzd[dxyzd>max_axis_lim] -= shift_position_constant
    dxyzd[dxyzd<min_axis_lim] += shift_position_constant
    return dxyzd
    
def dist_mag(dx,dy,dz):
    dist_m = np.sqrt(dx**2+dy**2+dz**2)
    return dist_m
    
def r_200_cal(distance):
    r_0 = 0.
    r_1 = 2.
    r_2 = 4.
    while True:
        index = np.array([i for i, j in enumerate(distance) if j <= r_1])
        #print index
        #print r_0, r_1, r_2
        mass = m_particle*len(index)
        den = 3.*mass/(4.*np.pi*(r_1**3))
        #print den
        if np.abs(den-200.*d_critical)<=tolerance:
            break
        if (r_1-r_0)<=tolerance_r:
            r_1 = r_0
            break
        if (den-200.*d_critical)>tolerance:
            r_0 = r_1
            r_1 = (r_1+r_2)/2.
        if (den-200.*d_critical)<-tolerance:
            r_2 = r_1
            r_1 = (r_0+r_1)/2.        
    return r_1, mass
        
def halo_profile(pos, cor_m, intervals, radius):
    
    dx = distance_calc(pos,cor_m,0)[0]
    dy = distance_calc(pos,cor_m,1)[0]
    dz = distance_calc(pos,cor_m,2)[0]
    dist_temp = dist_mag(dx,dy,dz)
    distance = dist_temp
    
    info = r_200_cal(distance)
    r_200 = info[0]
    m_halo = info[1]
    
    mass_profile = m_particle*np.histogram(dist_temp,bins = intervals)[0]
    density_profile = 3.*mass_profile/(4.*np.pi*(radius**3-intervals[:len(intervals)-1]**3))
    
    return mass_profile, density_profile, r_200, m_halo
    

fName = '/Users/Yuyu/Halo/GADGET/groups_catalogue/fof_special_catalogue_064'


aGroupLen, aGroupOffset, aGroupMass, aGroupCM, aGroupNspecies, aGroupSfr = read_groups_catalogue(fName)

fName1 = '/Users/Yuyu/Halo/GADGET/groups_particles/fof_special_particles_064'

aPos, aNtot = read_groups_particles(fName1)

fName2 = '/Users/Yuyu/Halo/GADGET/groups_indexlist/fof_special_indexlist_064'

ntot, idx = read_groups_indexlist(fName2)

  
#print aPos.shape
#print
#print aGroupMass.shape
#print aGroupLen.sum()
  
  
idx = [] 
for i in range(1, aGroupLen.size+1):
    
    idx = np.append(idx, i*np.ones(aGroupLen[i-1]).astype(int))
  
  
x, y, z = (aPos/1000)[:,0], (aPos/1000)[:,1], (aPos/1000)[:,2]
cor = np.array([list(t) for t in zip(x,y,z)])

groups_index = np.unique(idx, return_index=True, return_inverse=True, return_counts=True)

index_100 = np.array([i for i, j in enumerate(groups_index[3]) if j >= 100])

group_id = groups_index[0]
group_id = group_id[index_100]

group_size = groups_index[3]
group_size = group_size[index_100]

particle_start = groups_index[1]
particle_start = particle_start[index_100]

m_profile = np.zeros((len(group_id),len(radius)))
d_profile = np.zeros((len(group_id),len(radius)))
r_200 = np.zeros(len(group_id))
m_halo = np.zeros(len(group_id))

for i in range(len(group_id)):
    print i
    pos = cor[particle_start[i]:particle_start[i]+group_size[i]]
    cor_m = aGroupCM[i]/1000.
    profile = halo_profile(pos, cor_m, intervals, radius)
    
    m_profile[i] = profile[0]
    d_profile[i] = profile[1]
    r_200[i] = profile[2]
    m_halo[i] = profile[3]
    #print r_200[i], m_halo[i]
    
np.save('GADGET/fof-064-m_profile', m_profile)
np.save('GADGET/fof-064-d_profile', d_profile)
np.save('GADGET/fof-064-r_200', r_200)
np.save('GADGET/fof-064-m_200', m_halo)
