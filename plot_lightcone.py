import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm

#f =fits.util.get_testdata_filepath('/Users/Yuyu/kSZ/kSZ_Box0_35x35.fits')
hdulist = fits.open('/Users/Yuyu/kSZ/kSZ_Box0_35x35.fits', memmap=True)

hdulist.info()

b = hdulist[0].data

plt.figure()
plt.imshow(b, cmap='viridis')#,norm=LogNorm(vmin=np.min(b),vmax=np.max(b)))
plt.show()