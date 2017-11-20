import pynbody
import numpy as np


fileIn = '/Users/Yuyu/Halo/GADGET/snapshot_051'

s = pynbody.load(fileIn)
s.physical_units()

h = s.halos()