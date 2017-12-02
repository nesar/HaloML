import numpy as np
import matplotlib.pyplot as plt

bins=0.2
tbins=5.
n_bin = 40

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

cut = 0

plt.figure(100)
for h_num in range(50,150, 1):
    #plt.errorbar(radius, d_aver, yerr=d_std, color='r')
    plt.plot(radius[d_profile[h_num] > cut], d_profile[h_num][d_profile[h_num] > cut], 'ro-', alpha = 0.1, markersize = 3)
    #plt.scatter(radius, d_profile[h_num], color='r')
plt.xscale("log", nonposx='clip')
plt.yscale("log", nonposy='clip')


for h_num in range(350,450,1):
    #plt.errorbar(radius, d_aver, yerr=d_std, color='r')
    plt.plot(radius[d_profile[h_num] > cut], d_profile[h_num][d_profile[h_num] > cut], 'bo:', alpha = 0.1, markersize = 3)
    #plt.scatter(radius, d_profile[h_num], color='r')
plt.xscale("log", nonposx='clip')
plt.yscale("log", nonposy='clip')


for h_num in range(900,1000,1):
    #plt.errorbar(radius, d_aver, yerr=d_std, color='r')
    plt.plot(radius[d_profile[h_num] > cut], d_profile[h_num][d_profile[h_num] > cut], 'go--', alpha = 0.1, markersize = 3)
    #plt.scatter(radius, d_profile[h_num], color='r')
plt.xscale("log", nonposx='clip')
plt.yscale("log", nonposy='clip')

print('d_profile[50]')

plt.show()
