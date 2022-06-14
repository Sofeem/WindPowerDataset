# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:45:30 2021

@author: user
"""
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sb
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



Winddata = pd.read_csv('C:/Users/user/Desktop/Data_Minning_Project/two_years_merged_and_reduced.csv')
#print(Winddata.describe)

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


#Scatterplot for SE1 price region for overall records 

# fig, axs = plt.subplots(1, 6, sharey=True)
# WindSE1.plot.scatter(x = 'Temperature' , y = 'power-production', xlim = (200,350), ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'RelativeHumidity' , y = 'power-production', ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'Wind_U' , y = 'power-production',  ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'Wind_V' , y = 'power-production',  ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'CloudCover' , y = 'power-production',  ylim = (0,1500), s = 0.5)
# WindSE1.plot.scatter(x = 'WindGustSpeed' , y = 'power-production', ylim = (0,1500), s = 0.5)


#Scatterplot for SE1 price region for overall records 
fig, axs = plt.subplots(1, 7, sharey=True)
WindSE1.plot(kind='scatter', x='Temperature', y='power-production', ax=axs[0], figsize=(16, 6))
WindSE1.plot(kind='scatter', x='RelativeHumidity', y='power-production', ax=axs[1])
WindSE1.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[2])
WindSE1.plot(kind='scatter', x='Wind_V', y='power-production', ax=axs[3])
WindSE1.plot(kind='scatter', x='CloudCover', y='power-production', ax=axs[4])
WindSE1.plot(kind='scatter', x='WindGustSpeed', y='power-production', ax=axs[5])
WindSE1.plot(kind='scatter', x='Pressure', y='power-production', ax=axs[6])



#Finding corelation between variables 
#Price region SE1
pearsoncorrTempSE1 = WindSE1[['Temperature', 'power-production']].corr(method = 'pearson')
print(pearsoncorrTempSE1)
pearsoncorrRHSE1 = WindSE1[['RelativeHumidity', 'power-production']].corr(method = 'pearson')
print(pearsoncorrRHSE1)
pearsoncorrWUSE1 = WindSE1[['Wind_U', 'power-production']].corr(method = 'pearson')
print(pearsoncorrWUSE1)
pearsoncorrWVSE1 = WindSE1[['Wind_V', 'power-production']].corr(method = 'pearson')
print(pearsoncorrWVSE1)
pearsoncorrCCSE1 = WindSE1[['CloudCover', 'power-production']].corr(method = 'pearson')
print(pearsoncorrCCSE1)
pearsoncorrWGSSE1 = WindSE1[['WindGustSpeed', 'power-production']].corr(method = 'pearson')
print(pearsoncorrWGSSE1)
pearsoncorrPSE1 = WindSE1[['Pressure', 'power-production']].corr(method = 'pearson')
print(pearsoncorrPSE1)
# sb.heatmap(pearsoncorrSE1, annot=True)
# plt.show()



#Mormalize the dataset
WindSE1test = WindSE1.copy()
WindSE1test.drop(['time', 'cluster', 'region'],axis = 1, inplace=True)

# column_maxes = WindSE1test.max()
# df_max = column_maxes.max()
# column_mins = WindSE1test.min()
# df_min = column_mins.min()
# normalized_df = (WindSE1test - df_min) / (df_max - df_min)
for column in WindSE1test.columns:
    WindSE1test[column] = WindSE1test[column]  / WindSE1test[column].abs().max()
      
# view normalized data
#print(WindSE1test)

normalized_df = WindSE1test


#After normalized data
normalized_dfshort = normalized_df.head(75)

#normalized_dfshort.plot(kind='scatter', x='Temperature', y='power-production')
fig, axs = plt.subplots(1, 6, sharey=True)
normalized_dfshort.plot(kind='scatter', x='Temperature', y='power-production', ax=axs[0], figsize=(16, 6))
normalized_dfshort.plot(kind='scatter', x='RelativeHumidity', y='power-production', ax=axs[1])
normalized_dfshort.plot(kind='scatter', x='Wind_U', y='power-production', ax=axs[2])
normalized_dfshort.plot(kind='scatter', x='Wind_V', y='power-production', ax=axs[3])
normalized_dfshort.plot(kind='scatter', x='CloudCover', y='power-production', ax=axs[4])
normalized_dfshort.plot(kind='scatter', x='WindGustSpeed', y='power-production', ax=axs[5])


#Scatter plot with regressionLine
fig, axs = plt.subplots(1, 6,figsize=(16, 6), sharey=True)
fig.suptitle('Scatter Plot with Power production')
sb.regplot(x = "Temperature",
            y = "power-production", 
            ci = None,
            ax=axs[0],
            data = normalized_dfshort)


sb.regplot(x = "RelativeHumidity",
            y = "power-production", 
            ci = None,
            ax=axs[1],
            data = normalized_dfshort)
sb.regplot(x = "Wind_U",
            y = "power-production", 
            ci = None,
            ax=axs[2],
            data = normalized_dfshort)
sb.regplot(x = "Wind_V",
            y = "power-production", 
            ci = None,
            ax=axs[3],
            data = normalized_dfshort)
sb.regplot(x = "CloudCover",
            y = "power-production", 
            ci = None,
            ax=axs[4],
            data = normalized_dfshort)
sb.regplot(x = "WindGustSpeed",
            y = "power-production", 
            ci = None,
            ax=axs[5],
            data = normalized_dfshort)




# Linear Regression using sinle values for SE1 with Temperature
LRdatasetTempPP = normalized_df[['Temperature','power-production']]
XT = LRdatasetTempPP.iloc[:,:-1].values
yT = LRdatasetTempPP.iloc[:,1].values
XT_train, XT_test, yT_train, yT_test = train_test_split(XT, yT, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(XT_train, yT_train)
print(regressor.intercept_)
print(regressor.coef_)

yT_pred = regressor.predict(XT_test)

mseT = mean_squared_error(yT_test, yT_pred)
print('Temp MSE: ' + str(mseT))

RMSET = r2_score(yT_test, yT_pred)
print('Temp RMSE: ' + str(RMSET))

#Print 
df = pd.DataFrame({'Actual Power_production': yT_test.flatten(), 'Predicted Power_production': yT_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



#Linear Regression using sinle values for SE1 with RelativeHumidity 
LRdatasetRHPP = normalized_df[['RelativeHumidity','power-production']]
XRH = LRdatasetRHPP.iloc[:,:-1].values
yRH = LRdatasetRHPP.iloc[:,1].values
XRH_train, XRH_test, yRH_train, yRH_test = train_test_split(XRH, yRH, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(XRH_train, yRH_train)
print(regressor.intercept_)
print(regressor.coef_)

yRH_pred = regressor.predict(XRH_test)

mseRH = mean_squared_error(yRH_test, yRH_pred)
print('RH mse: ' + str(mseRH))

RMSERH = r2_score(yRH_test, yRH_pred)
print('RH RMSE: ' + str(RMSERH))


#Print 
df = pd.DataFrame({'Actual': yRH_test.flatten(), 'Predicted': yRH_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#Linear Regression using sinle values for SE1 with CloudCover
LRdatasetCloudCoverPP = normalized_df[['CloudCover','power-production']]
XCC = LRdatasetCloudCoverPP.iloc[:,:-1].values
yCC = LRdatasetCloudCoverPP.iloc[:,1].values
XCC_train, XCC_test, yCC_train, yCC_test = train_test_split(XCC, yCC, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(XCC_train, yCC_train)
print(regressor.intercept_)
print(regressor.coef_)

yCC_pred = regressor.predict(XCC_test)

mseCC = mean_squared_error(yCC_test, yCC_pred)
print('CC mse: ' + str(mseCC))

RMSECC = r2_score(yCC_test, yCC_pred)
print('CC RMSE: ' + str(RMSECC))

#Print 
df = pd.DataFrame({'Actual': yCC_test.flatten(), 'Predicted': yCC_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



#Linear Regression using sinle values for SE1 with Wind_U
LRdatasetWindUPP = normalized_df[['Wind_U','power-production']]
XWU = LRdatasetWindUPP.iloc[:,:-1].values
yWU = LRdatasetWindUPP.iloc[:,1].values
XWU_train, XWU_test, yWU_train, yWU_test = train_test_split(XWU, yWU, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(XWU_train, yWU_train)
print(regressor.intercept_)
print(regressor.coef_)

yWU_pred = regressor.predict(XWU_test)
mseWU = mean_squared_error(yWU_test, yWU_pred)
print('WU mse: ' + str(mseWU))
RMSEWU = r2_score(yWU_test, yWU_pred)
print('WU RMSE: ' + str(RMSEWU))

#Print 
df = pd.DataFrame({'Actual': yWU_test.flatten(), 'Predicted': yWU_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



#Linear Regression using sinle values for SE1 with Wind_V
LRdatasetWindVPP = normalized_df[['Wind_V','power-production']]
XWV = LRdatasetWindVPP.iloc[:,:-1].values
yWV = LRdatasetWindVPP.iloc[:,1].values
XWV_train, XWV_test, yWV_train, yWV_test = train_test_split(XWV, yWV, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(XWV_train, yWV_train)
print(regressor.intercept_)
print(regressor.coef_)

yWV_pred = regressor.predict(XWV_test)

mseWV = mean_squared_error(yWV_test, yWV_pred)
print('WV mse: ' + str(mseWV))

RMSEWV = r2_score(yWV_test, yWV_pred)
print('WV RMSE: ' + str(RMSEWV))


#Print 
df = pd.DataFrame({'Actual': yWV_test.flatten(), 'Predicted': yWV_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



#Linear Regression using sinle values for SE1 with WindGustSpeed
LRdatasetWindGustSpeedVPP = normalized_df[['WindGustSpeed','power-production']]
XWWGS = LRdatasetWindGustSpeedVPP.iloc[:,:-1].values
yWWGS= LRdatasetWindGustSpeedVPP.iloc[:,1].values
XWWGS_train, XWWGS_test, yWWGS_train, yWWGS_test = train_test_split(XWWGS, yWWGS, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(XWWGS_train, yWWGS_train)
print(regressor.intercept_)
print(regressor.coef_)

yWGS_pred = regressor.predict(XWWGS_test)

mseWGS = mean_squared_error(yWWGS_test, yWGS_pred)
print('WGS mse: ' + str(mseWGS))

RMSEWGS = r2_score(yWWGS_test, yWGS_pred)
print('WGS RMSE: ' + str(RMSEWGS))

#Print 
df = pd.DataFrame({'Actual': yWWGS_test.flatten(), 'Predicted': yWGS_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()





#Linear Regression using sinle values for SE1 with Pressure
LRdatasetPressureVPP = normalized_df[['Pressure','power-production']]
XP = LRdatasetPressureVPP.iloc[:,:-1].values
yP= LRdatasetPressureVPP.iloc[:,1].values
XP_train, XP_test, yP_train, yP_test = train_test_split(XP, yP, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(XP_train, yP_train)
print(regressor.intercept_)
print(regressor.coef_)

yP_pred = regressor.predict(XP_test)

mseP = mean_squared_error(yP_test, yP_pred)
print('P mse: ' + str(mseP))

RMSEP = r2_score(yP_test, yP_pred)
print('P RMSE: ' + str(RMSEP))

#Print 
df = pd.DataFrame({'Actual': yP_test.flatten(), 'Predicted': yP_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()





