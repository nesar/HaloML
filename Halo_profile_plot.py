import numpy as np
import matplotlib.pyplot as plt

bins=0.2
tbins=1.5
n_bin = 30

intervals = np.logspace(np.log10(bins), np.log10(tbins), num = n_bin)
radius = intervals[1:]

#intervals = np.array([nn*bins for nn in range(1,int(tbins/bins)+1)])
#radius = intervals[1:]


m_profile = np.load('GADGET/fof-064-m_profile.npy')
d_profile = np.load('GADGET/fof-064-d_profile.npy')

m_aver = np.average(m_profile, axis=0)
d_aver = np.average(d_profile, axis=0)

m_std = np.std(m_profile, axis=0)
d_std = np.std(d_profile, axis=0)

#h_num = 1

#plt.figure()
##plt.errorbar(radius, m_aver, yerr=m_std, color='b')
#plt.plot(radius, m_profile[h_num], color='b')
#plt.scatter(radius, m_profile[h_num], color='b')
#plt.xscale("log", nonposx='clip')
#plt.yscale("log", nonposy='clip')
#plt.show()

d_critical = 1.3598e11
cut = 0

plt.figure()
for h_num in range(900,1100):
    #plt.errorbar(radius, d_aver, yerr=d_std, color='r')
    dd = d_profile[h_num]
    plt.plot(radius[dd> cut], dd[dd> cut]/d_critical)
    #plt.scatter(radius, d_profile[h_num], color='r')
plt.xscale("log", nonposx='clip')
plt.yscale("log", nonposy='clip')
plt.show()

print('d_profile[50]')



## --------------- i/o export parameter -- just mass right now --------------
d_profile_list = []
for h_num in range(0,1000,1):
    d_profile_list.append(d_profile[h_num][d_profile[h_num] > cut].tolist())

print(d_profile_list)
# array.insert(0,var)

import json
with open('density_profile.txt', 'w') as outfile:
    json.dump(d_profile_list, outfile)


# halo_para = (1e10*np.abs(np.random.standard_normal(1000))).tolist()
halo_para = np.random.randint(10, size = 1000).tolist()
d_profile = np.load('GADGET/fof-064-m_200.npy').tolist()
with open('halo_parameters.txt', 'w') as outfile:
    json.dump(halo_para, outfile)




