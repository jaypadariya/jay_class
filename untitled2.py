from google.colab import drive
#drive.mount('/content/gdrive')

import matplotlib.pyplot as plt
import pandas as pd

wheat_data = pd.read_csv(r'C:\Users\student\rice_paddy.csv')

wheat_data.info()

wheat_data.head()

 wheat = wheat_data.groupby('Year')

wheat_data = wheat_data.set_index('Year')
wheat_data.index

y = wheat_data['Production in Tonne']

y.plot(figsize=(15, 6))
plt.ylabel("production in tonne")
plt.title("Produciton of wheat from 1961 - 2017")
plt.show()

import seaborn as sns

sns.set()

wheat_data.plot()

"""Trend in Area Harvested and Production"""

area = wheat_data[['Area harvested']]
area.rolling(12).mean().plot(figsize=(5,5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.title("Area harvested")
plt.show()

production = wheat_data[['Production in Tonne']]
production.rolling(12).mean().plot(figsize=(5,5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.title("Production in Tonne")
plt.show()

"""Seasonality in data"""

production.diff().plot(figsize=(5,5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.show()

area.diff().plot(figsize=(5,5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

# co-relation Linear regression
sns.lmplot(x='Area harvested', y='Yield(hg/ha)', fit_reg=True, data=wheat_data);

#polynomial regression
sns.lmplot(x='Area harvested', y='Yield(hg/ha)', order = 4, fit_reg=True, data=wheat_data);

sns.lmplot(x='Area harvested', y='Production in Tonne', fit_reg=True, data=wheat_data);

sns.lmplot(x='Year Code', y='Production in Tonne', fit_reg=True, data=wheat_data);

# sns.lmplot(x='Year Code', y='Production in Tonne', fit_reg=False, data=wheat_data, hue='Yield(hg/ha)');
# !pip install pyEX



"""Moving Average"""

df=pd.DataFrame(wheat_data)
ts = pd.Series(df["Yield(hg/ha)"].values, index=df["Year Code"])
# print(ts.head(5))
mean_smoothed = ts.rolling(window=5).mean()
# print(mean_smoothed)
###### NEW #########
# mean_smoothed[0]=ts[0]
# mean_smoothed.interpolate(inplace=True)
####################
exp_smoothed = ts.ewm(alpha=0.5).mean()

h1 = ts.head(8)
h2 = mean_smoothed.head(8)
h3 = exp_smoothed.head(8)
k = pd.concat([h1, h2, h3], join='outer', axis=1)
k.columns = ["Actual", "Moving Average", "Exp Smoothing"]
print(k)


plt.figure(figsize=(16,5))
plt.plot(ts, label="Original")
plt.plot(mean_smoothed, label="Moving Average")
plt.plot(exp_smoothed, label="Exponentially Weighted Average")
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# forecast error using polynomial regression 

# transforming the data to include another axis
x = wheat_data['Area harvested'][:, np.newaxis]
y = wheat_data['Yield(hg/ha)'][:, np.newaxis]

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print("RMSE",rmse)
print("R2",r2)
