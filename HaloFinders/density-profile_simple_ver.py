#********** EVAPORATING_DM check **************
# AHF   manually compute density profiles
#Keita Todoroki
#8/11/2015

from readgadget import *
from numpy import *
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt

import matplotlib.colors as colors
import matplotlib.cm as cmx


r,m,mg,ms,den     =genfromtxt('/Users/Physics/Desktop/Series/S246/DM_profile/snap4.0000.00000.AHF_profiles',usecols=(0,2,2,2,4),unpack=True)
mv,rv,vmax        =genfromtxt('/Users/Physics/Desktop/Series/S246/DM_profile/snap4.0000.00000.AHF_halos',usecols=(8,9,10),unpack=True)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

fig.patch.set_facecolor('white')

ax.set_xlabel(r'R [kpc]',fontsize=15)
ax.set_ylabel(r'Density [M$_{\odot}$ pc$^{-3}$]',fontsize=15)


for i in range(len(mv)):
    rv = rv[i]
    mv = mv[i]
    vmax = vmax[i]
    break


factor = 4.*pi/3.
Rho,R = [],[]
M = []

###

h = 0.67
unitconv = 1e9



ncolor = len(r)
color = cm.rainbow(linspace(0,1,ncolor))

N = 0


FLAG = 0


r = r / h
m = m / h

for i,c in zip(range(len(r)), color):
    #print i
    if (r[i] > 0):     # this is to exclude the Amiga's 'negative' inner radial range
        if (r[i+1]-r[i]) > 0:
            N = N + 1;
            r_mid = (r[i+1] - r[i])/2 + r[i]
            vol1 = factor * pow(r[i+1],3)
            vol2 = factor * pow(r[i],3)
            dvol = vol1 - vol2
            #dM = abs((m[i+1] - m[i]) - (mg[i+1] - mg[i]))
            dM = abs((m[i+1] - m[i]))
            rho = dM/dvol
            if rho > 0:
                Rho.append(rho / unitconv)
                R.append(r_mid)
        else:
            #plot(R,Rho,lw=4,color='gray',label='CDM')
            #print N
            FLAG = FLAG + 1
            if FLAG == 5:
                #savetxt('CDMrrho.txt',c_[R,Rho])
                savetxt('logrrho246.txt',c_[log10(R),log10(Rho)])
                #del Rho[:],R[:] # empty the list
                break









#plot(R,Rho,lw=1.5,color='k',label='a')
#plot(r,den,lw=1.5,color='r',label='b')




plt.xscale('log')
plt.yscale('log')


minorticks_on()
grid()
#plt.legend(loc='best',prop={'size':18}, fontsize=18).draw_frame(False)

show()


