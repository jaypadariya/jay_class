import keras
import numpy

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
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Users\student\Soil_Exam.csv")
X = df[['Longitude',"Latitude"]]
Y = df["EC"]
x=np.array(X[:])
y=np.array(Y[:])
x=x.reshape(-1,2)
y=y.reshape(-1,1)

df1 = pd.read_csv(r"C:\Users\student\Downloads\Soil_Eam_test.csv")
X1 = df[['Longitude',"Latitude"]]
Y1 = df["EC"]
x1=np.array(X1[:])
y1=np.array(Y1[:])
x1=x1.reshape(-1,2)
y1=y1.reshape(-1,1)

X_train, X_test= train_test_split(x,x1)
Y_train, Y_test = train_test_split(y,y1)
X_train=X_train.reshape(-1,2)
Y_train=Y_train.reshape(-1,1)
X_test=X_test.reshape(-1,2)
Y_test=Y_test.reshape(-1,1)
lm=LinearRegression().fit(X_train,Y_train)
predy=lm.predict(X_test)
plt.scatter(Y_test,predy)
plt.scatter(y,Y)
r2_score(predy,Y_test)


Scalerx,Scalery=MinMaxScaler(),MinMaxScaler()  #scaling the x and y coz the values we have taken is very high
Scalerx.fit(x)
Scalery.fit(y.reshape(10,1))  #it have to reshape coz complete 2d matrix is not y..it may have to make([100,1)
X=Scalerx.transform(x)   #define to one variable that can see in variable exploral
Y=Scalery.transform(y.reshape(10,1)) 
#Y1=Scalery.transform(y1.reshape(10,1)) 

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse',optimizer='adam') #cross antropy is used in classification as a optimizer
model.fit(X,Y,epochs=5,verbose=0)
accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))








X,Y=make_regression(n_samples=100,n_features=2,noise=0.1,random_state=1) #for taking random value
ScalerX,ScalerY=MinMaxScaler(),MinMaxScaler()  #scaling the x and y coz the values we have taken is very high
ScalerX.fit(X)
ScalerY.fit(Y.reshape(100,1))  #it have to reshape coz complete 2d matrix is not y..it may have to make([100,1)
X=ScalerX.transform(X)   #define to one variable that can see in variable exploral
Y=ScalerY.transform(Y.reshape(100,1))  #it is define to see the reshaped y

model = Sequential()  #cal the sequential model
model.add(Dense(4, input_dim=2, kernel_initializer='normal', activation='relu'))  #(12 is the neuron,input dimension is 2-D )
model.add(Dense(4, activation='relu'))  #again we take 4 neaurons and relu activation function
model.add(Dense(1, activation='linear'))  #output layer which has alwys linear function and nural is alwys 1
model.compile(loss='mse',optimizer='adam') #cross antropy is used in classification as a optimizer
model.fit(X,Y,epochs=5,verbose=0)  #fit the model from X and Y

Xnew,a=make_regression(n_samples=3,n_features=2,noise=0.1,random_state=1) #again define values randomly
Xnew=ScalerX.transform(Xnew) #xnew shows negative vlaurs so it may have to scale in 0 to 1
Ynew=model.predict(Xnew) #define xnew in ynew for show
a=ScalerY.transform(a.reshape(3,1)) #reshape and transform a with yscaler it make 2d array
#mean scale error 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(a, Ynew) # we can make mean square error of a and Ynew
