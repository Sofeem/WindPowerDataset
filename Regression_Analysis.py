# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:45:30 2021

@author: user
"""
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



Winddata = pd.read_csv('G:/My Drive/PhD Program/Courses/Data Minning Project/Data_Minning_Project/two_years_merged_and_reduced.csv')
print(Winddata.describe)

#Groupby Region 

# multiple line plots
# plt.plot( 'time', 'power-production', data=Winddata, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
# plt.plot( 'time', 'power-production', data=Winddata, marker='', color='olive', linewidth=2)
# plt.plot( 'time', 'power-production', data=Winddata, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
# # show legend
# plt.legend()

# # show graph
# plt.show()



#Creating Subset of All data
WindSE1 = Winddata[Winddata ['region'] == 'SE1']
WindSE2 = Winddata[Winddata ['region'] == 'SE2']
WindSE3 = Winddata[Winddata ['region'] == 'SE3']
WindSE4 = Winddata[Winddata ['region'] == 'SE4']


#Scatterplot for SE1 price region for overall records 

# fig, axs = plt.subplots(1, 6, sharey=True)
# WindSE1.plot.scatter(x = 'Temperature' , y = 'power-production', xlim = (200,350), ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'RelativeHumidity' , y = 'power-production', ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'Wind_U' , y = 'power-production',  ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'Wind_V' , y = 'power-production',  ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'CloudCover' , y = 'power-production',  ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'WindGustSpeed' , y = 'power-production', ylim = (0,1500), s = 0.5)


#Scatterplot for SE1 price region for overall records 
fig, axs = plt.subplots(1, 6, sharey=True)
WindSE1.plot(kind='scatter', x='Temperature', y='power-production', ax=axs[0], figsize=(16, 6))
WindSE1.plot(kind='scatter', x='RelativeHumidity', y='power-production', ax=axs[1])
WindSE1.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[2])
WindSE1.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[3])
WindSE1.plot(kind='scatter', x='CloudCover', y='power-production', ax=axs[4])
WindSE1.plot(kind='scatter', x='WindGustSpeed', y='power-production', ax=axs[5])

#Scatterplot for SE2 price region for overall records 
fig, axs = plt.subplots(1, 6, sharey=True)
WindSE2.plot(kind='scatter', x='Temperature', y='power-production', ax=axs[0], figsize=(16, 6))
WindSE2.plot(kind='scatter', x='RelativeHumidity', y='power-production', ax=axs[1])
WindSE2.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[2])
WindSE2.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[3])
WindSE2.plot(kind='scatter', x='CloudCover', y='power-production', ax=axs[4])
WindSE2.plot(kind='scatter', x='WindGustSpeed', y='power-production', ax=axs[5])

#Scatterplot for SE3 price region for overall records 
fig, axs = plt.subplots(1, 6, sharey=True)
WindSE3.plot(kind='scatter', x='Temperature', y='power-production', ax=axs[0], figsize=(16, 6))
WindSE3.plot(kind='scatter', x='RelativeHumidity', y='power-production', ax=axs[1])
WindSE3.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[2])
WindSE3.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[3])
WindSE3.plot(kind='scatter', x='CloudCover', y='power-production', ax=axs[4])
WindSE3.plot(kind='scatter', x='WindGustSpeed', y='power-production', ax=axs[5])

#Scatterplot for SE4 price region for overall records 
fig, axs = plt.subplots(1, 6, sharey=True)
WindSE4.plot(kind='scatter', x='Temperature', y='power-production', ax=axs[0], figsize=(16, 6))
WindSE4.plot(kind='scatter', x='RelativeHumidity', y='power-production', ax=axs[1])
WindSE4.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[2])
WindSE4.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[3])
WindSE4.plot(kind='scatter', x='CloudCover', y='power-production', ax=axs[4])
WindSE4.plot(kind='scatter', x='WindGustSpeed', y='power-production', ax=axs[5])

#Finding corelation between variables 
#Price region SE1
pearsoncorrSE1 = WindSE1.corr(method = 'pearson')
print(pearsoncorrSE1)
#Price region SE2
pearsoncorrSE2 = WindSE2.corr(method = 'pearson')
print(pearsoncorrSE2)
#Price region SE3
pearsoncorrSE3 = WindSE3.corr(method = 'pearson')
print(pearsoncorrSE3)
#Price region SE4
pearsoncorrSE4 = WindSE4.corr(method = 'pearson')
print(pearsoncorrSE4)











# Linear Regression using sinle values for SE1 with Temperature

LRdatasetTempPP = WindSE1[['Temperature','power-production']]
XT = LRdatasetTempPP.iloc[:,:-1].values
yT = LRdatasetTempPP.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(XT, yT, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(mse)

RMSE = r2_score(y_test, y_pred)
print(RMSE)


#Print 
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Linear Regression using sinle values for SE1 with Temperature

LRdatasetTempPP = WindSE1[['Temperature','power-production']]
XT = LRdatasetTempPP.iloc[:,:-1].values
yT = LRdatasetTempPP.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(XT, yT, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(mse)

RMSE = r2_score(y_test, y_pred)
print(RMSE)


#Print 
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#Linear Regression using sinle values for SE1 with RelativeHumidity 

#Linear Regression using sinle values for SE1 with CloudCover

#Linear Regression using sinle values for SE1 with RelativeHumidity 







