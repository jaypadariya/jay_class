
from osgeo import gdal
#gdalinfo image.tif
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import parser, rrule
from datetime import datetime, time, date
import datetime as tm
import calendar
import seaborn as sns 
import math
from sklearn import datasets
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"C:\Users\student\dk1.csv")
img=gdal.Open(r"C:\Users\student\2014_JAN_1_NDVI_MODIS.tif")
y=img.GetRasterBand(1)
f=y.ReadAsArray()
plt.imshow(f)

#from pixel value import lagitude longitude

from osgeo import ogr, gdal, osr
import numpy as np


raster = gdal.Open(r"C:\Users\student\2014_JAN_1_NDVI_MODIS.tif")
raster_array = np.array(raster.ReadAsArray())
y=raster.GetRasterBand(1)
f=y.ReadAsArray()
plt.imshow(f)
X=7200
Y=3600
def pixel2coord(x,y):
   for(x,y) in range (X,Y):
     xoff, a, b, yoff, d, e = raster.GetGeoTransform()
     xp = a * x + b * y + xoff
     yp = d * x + e * y + yoff
     return(xp, yp)

print(pixel2coord(xp,yp))