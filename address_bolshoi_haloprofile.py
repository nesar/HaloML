import numpy as np

def find_closest(alist, target):
    return min(alist, key=lambda x:abs(x-target))

def list_matching(list1, list2):
    list1_copy = list1[:]
    pairs = []
    for i, e in enumerate(list2):
        elem = find_closest(list1_copy, e)
        pairs.append([i, list1.index(elem)])
        list1_copy.remove(elem)
    return np.array(pairs)
    
def diff_replace(same,full):
    diff_l = set(full)-set(same)
    diff = list(diff_l)
    
    diff_rep=[]
    for v in diff:
        while v not in same and v!=0:
            v -= 1
        diff_rep.append(v)
    
    return diff, diff_rep

full_len = range(30)
halo_profile = np.zeros((1000,30))
halo_r=np.zeros(1000)

for ix in range(1000):
    filenum=str(ix)
    #print filenum
    data=[]
    halo=[]
    f1=open("/Users/Yuyu/Halo/Bolshoi/halo_profile/BDM-Halo-"+str(filenum)+".csv","r")
    while 1:
        rawline = f1.readline()
        lines = rawline.strip()
        if lines == '':
            break
        if lines[1] != "r":
            line = map(float, lines.split(','))
            rbin = line[6]
            profile = line[9]
            data.append(rbin)
            halo.append(profile)
    f1.close()
    
    #Find the same index of Bolshoi radius bin and the full radius bin
    intervals = np.logspace(np.log10(data[-1]/28.), np.log10(data[-1]), num = 30)
    halo_r[ix] = intervals[23]
    index_information = list_matching(intervals.tolist(), data)
    halo_profile[ix,index_information[:,1]] = np.array(halo)
            
    if len(data)!=30:
        diff, diff_rep = diff_replace(index_information[:,1],full_len)
        #print halo_profile[ix]
        halo_profile[ix,diff] = halo_profile[ix,diff_rep]
        #print 30-len(data), diff, diff_rep, halo_profile[ix]
        #print "====================================================="
            
np.save("/Users/Yuyu/Halo/Bolshoi/Bolshoi_All_density_profile.npy", halo_profile)
np.save("/Users/Yuyu/Halo/Bolshoi/Bolshoi_Radius.npy", halo_r)
