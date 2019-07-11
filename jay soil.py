from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

file_train=pd.read_csv(r"C:\Users\student\Soil_Exam.csv")
file_test=pd.read_csv(r"C:\Users\student\Downloads\Soil_Eam_test.csv")

x_train=file_train[['Latitude','Longitude']]
y_train=file_train[['EC']]

x_test=file_test[['Latitude','Longitude']]
y_test=file_test[['EC']]





model=Sequential()
model.add(Dense(10,input_dim=2,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse',optimizer='adam')


model.fit(x_train,y_train,epochs=90,verbose=0)

y_predict=model.predict(x_test)
mse=math.sqrt(mean_squared_error(y_test,y_predict))
print(mse)





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

