# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:24:39 2019

@author: student
"""

import numpy as np
from osgeo import gdal
from osgeo import ogr
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
x=[2464,2630,2832,2642,2607,2675,2557,2855,2887]
t=[1,2,3,4,5,6,7,8,9]
t=np.array(t)
x=np.array(x)
t=t.reshape(-1,1)
x=x.reshape(-1,1)
plt.scatter(t,x)
regr=LinearRegression()
regr.fit(t,x)


x_pred=regr.predict(t)
print('R2_score',r2_score(x,x_pred))
print('Coefficients: \n', regr.coef_)
plt.scatter(t,x)
plt.plot(t,x_pred)
