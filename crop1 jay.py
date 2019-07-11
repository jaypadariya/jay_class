# task 1: visualize a satellite image and shapefile using python
# task 2: descriptive analysis and acreage calculation

# import required libraries

from osgeo import gdal
from osgeo import ogr
import numpy as np
import math
from scipy.stats import mode
import os
import matplotlib.pyplot as plt



img = gdal.Open(r'C:\Users\student\1.tif')
band=img.GetRasterBand(1)
rast_array=band.ReadAsArray()
plt.imshow(rast_array)


rast_array = np.array(band.ReadAsArray())
rast_array .flatten()


min=np.min(rast_array)
max=np.max(rast_array)
meanval = np.mean(rast_array)
medianval = np.median(rast_array)
    
sdval = np.std(rast_array)
varianceval = np.var(rast_array)
rangeval = max - min

print("mean:",meanval)   
print("medain:",medianval)
print("sdval:",sdval)
print("varia:",varianceval)
print("range:",rangeval)


plt.hist(rast_array)



