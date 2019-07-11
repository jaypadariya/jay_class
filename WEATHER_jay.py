from osgeo import gdal
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
df = pd.read_csv(r"C:\Users\student\INDIA105_weather1.csv")
df=df[["TemperatureC","PressurehPa","Humidity","HourlyPrecipMM","dailyrainMM","soalr"]]
X = df['PressurehPa']
Y = df["TemperatureC"]
x=np.array(X[:])
y=np.array(Y[:])

x=x.reshape(-1,1)
y=y.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=1)
X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
Y_test=Y_test.reshape(-1,1)




N = 3

def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements
#derive_nth_day_feature(df,"TemperatureC",3)

for feature in ["TemperatureC","PressurehPa","Humidity","HourlyPrecipMM","dailyrainMM","soalr"]:
   # if feature != 'day':
        for N in range(1, 5):
            derive_nth_day_feature(df, feature, N) 
            
            
lm=LinearRegression().fit(X_train,Y_train)
predy=lm.predict(X_test)         
            
r2_score(Y_test,predy)
print(r2_score)
# t-test
from scipy import stats
t2, p2 = stats.ttest_ind(X,Y)
print("t-test", t2)


# f-test:

import statistics as stats
import scipy.stats as ss
d1=df['PressurehPa',"Humidity","HourlyPrecipMM","dailyrainMM","soalr"]
d2=df["TemperatureC"]
def Ftest_pvalue(d1,d2):
    df1 = len(d1) - 1
    df2 = len(d2) - 1
    F = stats.variance(d1) /stats.variance(d2)
    single_tailed_pval = ss.f.cdf(F,df1,df2)
    double_tailed_pval = single_tailed_pval * 2
    return double_tailed_pval
    print("F-test value:",Ftest_pvalue( d1,d2))
ftestvalue=print("F-test value:",Ftest_pvalue( d1,d2))    
            
            
#corelation with others
df.corr()[["TemperatureC"]].sort_values("TemperatureC")
predictors = ['Humidity_1','Humidity_2','Humidity','Humidity_3''soalr_1','soalr_2','soalr','soalr_3','TemperatureC_3','TemperatureC_2','TemperatureC_1']
new_Data = df[['TemperatureC'] + predictors] 
#




    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

#derive_nth_day_feature(df,"TemperatureC",3)

for feature in ["TemperatureC","PressurehPa","Humidity","HourlyPrecipMM","dailyrainMM","soalr"]:
   # if feature != 'day':
        for N in range(1, 5):
            derive_nth_day_feature(df, feature, N)             
       
        
        
        
        
        
        
        
        
from sklearn import linear_model
X = new_Data[['TemperatureC_3','TemperatureC_2','TemperatureC_1']]
Y = new_Data[['TemperatureC']]

xx = X.ix[5:]
yy = Y.ix[5:]
regr = linear_model.LinearRegression()
regr.fit(xx,yy)

127.# Make predictions using the testing set
y_pred = regr.predict(xx)        
            
print('Coefficients: \n', regr.coef_)             
print("Mean squared error: %.2f" )
print('Variance score: %.2f' % r2_score(yy, y_pred)) 
plt.scatter(yy,y_pred)