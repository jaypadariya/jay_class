import matplotlib.pyplot as plt
import pandas as pd
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
df = pd.read_csv(r'C:\Users\student\rice_paddy.csv')
df1=df["Yield in Tonne"]
df2=df["Production in Tonne"]
df3=df["Year"]
plt.plot(df2)
sns.set()
df2.plot()
df1.plot()
area = df[['Area harvested']]
area.rolling(12).mean().plot(figsize=(5,5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.title("Area harvested")
plt.show()


production = df[['Production in Tonne','Area harvested']]
production.rolling(12).mean().plot(figsize=(4,4), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.title("Production in Tonne")
plt.show()


sns.lmplot(x='Production in Tonne',y="Year",data=df)
production.diff().plot(figsize=(4,5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.show()

area.diff().plot(figsize=(4,5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
