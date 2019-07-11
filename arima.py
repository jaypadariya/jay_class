#Arima


import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
%matplotlib inline
import itertools

plt.style.use('bmh')
data_matrix = pd.read_csv(r"C:\Users\student\LaturRains_1965_2002 (1).csv",sep="\t")
type(data_matrix)
data_matrix.head()
data_matrix.set_index('Year', inplace=True)
data_matrix.tail()
data_matrix = data_matrix.transpose()
data_matrix
dates = pd.date_range(start='1965-01', freq='MS', periods=len(data_matrix.columns)*12)
dates
plt.figure(figsize=(5,5))
plt.plot(data_matrix)
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Month vs Precipitation across all years')
plt.boxplot(data_matrix.mean(axis=0))
plt.xlabel('Month')
plt.ylabel('Montly Mean Precipitation(mm)')
plt.title('Month vs Precipitation across all years')
rainfall_data_matrix_np = data_matrix.transpose().as_matrix()
# rainfall_data_matrix_np.shape
shape = rainfall_data_matrix_np.shape
rainfall_data_matrix_np = rainfall_data_matrix_np.reshape((shape[0] * shape[1], 1))
rainfall_data_matrix_np.shape
rainfall_data = pd.DataFrame({'Precipitation': rainfall_data_matrix_np[:,0]})
rainfall_data.set_index(dates, inplace=True)

test_data = rainfall_data.ix['1995': '2002']
train_data = rainfall_data.ix[: '1994']
train_data.tail() # 1965-1994
test_data.tail() # 1995-2002
plt.figure(figsize=(5,5))
plt.plot(rainfall_data, color='blue')
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation in mm')


rainfall_data_matrix_np = data_matrix.transpose().as_matrix()
# rainfall_data_matrix_np.shape
shape = rainfall_data_matrix_np.shape
rainfall_data_matrix_np = rainfall_data_matrix_np.reshape((shape[0] * shape[1], 1))
rainfall_data_matrix_np.shape



###
fig, axes = plt.subplots(2, 2, sharey=False, sharex=False)
fig.set_figwidth(14)
fig.set_figheight(8)
axes[0][0].plot(rainfall_data.index, rainfall_data, label='Original')
axes[0][0].plot(rainfall_data.index, rainfall_data.rolling(window=4).mean(), label='4-Months Rolling Mean')
axes[0][0].set_xlabel("Years")
axes[0][0].set_ylabel("Precipitation in mm")
axes[0][0].set_title("4-Months Moving Average")
axes[0][0].legend(loc='best')
############
axes[0][1].plot(rainfall_data.index, rainfall_data, label='Original')
axes[0][1].plot(rainfall_data.index, rainfall_data.rolling(window=8).mean(), label='8-Months Rolling Mean')
axes[0][1].set_xlabel("Years")
axes[0][1].set_ylabel("Precipitation in mm")
axes[0][1].set_title("8-Months Moving Average")
axes[0][1].legend(loc='best')
############
axes[1][0].plot(rainfall_data.index, rainfall_data, label='Original')
axes[1][0].plot(rainfall_data.index, rainfall_data.rolling(window=12).mean(), label='12-Months Rolling Mean')
axes[1][0].set_xlabel("Years")
axes[1][0].set_ylabel("Precipitation in mm")
axes[1][0].set_title("12-Months Moving Average")
axes[1][0].legend(loc='best')
############
axes[1][1].plot(rainfall_data.index, rainfall_data, label='Original')
axes[1][1].plot(rainfall_data.index, rainfall_data.rolling(window=16).mean(), label='16-Months Rolling Mean')
axes[1][1].set_xlabel("Years")
axes[1][1].set_ylabel("Precipitation in mm")
axes[1][1].set_title("16-Months Moving Average")
axes[1][1].legend(loc='best')
# ############
plt.tight_layout()
plt.show()



####
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
adf_test(rainfall_data.Precipitation) 






#lstm
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_matrix = pd.read_csv(r"C:\Users\student\LaturRains_1965_2002 (1).csv",sep="\t")
type(data_matrix)
data_matrix.set_index('Year', inplace=True)
data_matrix = data_matrix.transpose()
data_matrix.head()
dates = pd.date_range(start='1965-01', freq='MS', periods=len(data_matrix.columns)*12)
dates
rainfall_data_matrix_np = data_matrix.transpose().as_matrix()

shape = rainfall_data_matrix_np.shape
rainfall_data_matrix_np = rainfall_data_matrix_np.reshape((shape[0] * shape[1], 1))
rainfall_data = pd.DataFrame({'Precipitation': rainfall_data_matrix_np[:,0]})
rainfall_data.set_index(dates, inplace=True)

plt.figure(figsize=(20,5))
plt.plot(rainfall_data, color='blue')
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation in mm')
test_data = rainfall_data.ix['1995': '2002']
train_data = rainfall_data.ix[: '1994']
type(train_data)
train_data.tail() # 1965-1994
test_data.head() # 1995-2002
plt.figure(figsize=(20,5))
plt.plot(rainfall_data, color='blue')
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation data in mm of Latur from 1965-2002')
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from keras.callbacks import LambdaCallback

data_raw = rainfall_data.values.astype("float32")

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(data_raw)

# Print a few values.
dataset[0:5]
TRAIN_SIZE = 0.8
train_size = int(len(dataset)*TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print (dataset.shape)
print (train.shape)
print (test.shape)
print("Number of entries (training set, test set): " + str((len(train), len(test))))
def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
#         print("--",dataset[i+window_size,0],"--")
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))
    # Create test and training sets for one-step-ahead regression.
window_size = 12
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
# print(train_Y)
print(train_X.shape)
print(train_Y.shape)

# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
train_Y = np.reshape(train_Y, (train_Y.shape[0], 1))
test_Y = np.reshape(test_Y, (test_Y.shape[0], 1))
print("New training data shape:")
print(train_X.shape)
print(train_Y.shape)
# train_X[:5]
def fit_model(train_X,train_Y,window_size=1):
    
    model = Sequential()
    model.add(LSTM(6,input_shape=(1,window_size)))
#     model.add(LSTM(6,input_shape=(1,window_size)))
    model.add(Dense(1))
#     print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
    model.compile(loss='mean_squared_error',
                 optimizer='adam', metrics=['mape', 'accuracy'])
#     history = model.fit(train_X,train_Y,validation_data=(test_X, test_Y),epochs=250,batch_size=1,callbacks = [print_weights],verbose=2)
    history = model.fit(train_X,train_Y,validation_data=(test_X, test_Y),epochs=250,batch_size=1,verbose=2)
    return model,history

#     on_epoch_end=lambda batch, logs: print (model.layers[1].get_weights())
#fit the model
model1, history = fit_model(train_X, train_Y, window_size)
plot_model(model1, to_file='LSTM_Latur_plot_1.png', show_shapes=True, show_layer_names=True)
model1.summary()
train_Y.shape
test_Y.shape
def predict_and_score(model,X,Y):
    #Make predictions on the original scale of data
    pred = scaler.inverse_transform(model.predict(X))
    #Prepare Y also to be in original data scale
    orig_data = scaler.inverse_transform(Y)
#     print(orig_data)
#     print("-----")
#     print(pred[:,0])
    #Calculate RMSE
    score = math.sqrt(mean_squared_error(orig_data, pred[:, 0]))
    return (score,pred)

train_rmse, train_predict = predict_and_score(model1, train_X, train_Y)
test_rmse, test_predict = predict_and_score(model1, test_X, test_Y)

# train_rmse, train_predict = predict_and_score(model1, train_X, np.reshape(train_Y, (train_Y.shape[0], 1,1)))
# test_rmse, test_predict = predict_and_score(model1, test_X, np.reshape(test_Y, (test_Y.shape[0],1, 1)))

print("Training data score: %.2f RMSE" % train_rmse)
print("Test data score: %.2f RMSE" % test_rmse)

# start with training predictions
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:,:] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict
