# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:47:30 2019

@author: GeoSpatial-3
"""

Lat1 = np.array(Lat)
Lon1 = np.array(Lon)
#Lat = input('Enter the Valid Lat Value : ')
#Lon = input('Enter the Valid LON Value :' )

#a = float(Lat) #24.26
#b = float(Lon) #72.3894
points = [(Lon1[0:1862], Lat1[0:1862])]

# open the raster file
ds = gdal.Open('SoilPM.tif')
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()

#if ds is None:
#   print 'Could not open the raster file'
#    sys.exit(1)
#else:
#    print 'The raster file was opened satisfactorily'

# get georeference info
transform = ds.GetGeoTransform() # (-2493045.0, 30.0, 0.0, 3310005.0, 0.0, -30.0)
xOrigin = transform[0] # -2493045.0
yOrigin = transform[3] # 3310005.0
pixelWidth = transform[1] # 30.0
pixelHeight = transform[5] # -30.0

band = ds.GetRasterBand(1) # 1-based index

data = band.ReadAsArray()

# loop through the coordinates
for point in points:
    x = point[0]
    y = point[1]

    xOffset = np.floor((x - xOrigin) / pixelWidth)
    yOffset = np.floor((y - yOrigin) / pixelHeight)
    value = data[yOffset.astype(int),xOffset.astype(int)]
    print (xOffset)
    print (yOffset)
    print (value)
    
     