# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:25:44 2021

@author: user
"""
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.regression.linear_model as sm
from sklearn.metrics import mean_squared_error
from math import sqrt


Winddata = pd.read_csv('C:/Users/user/Desktop/Data_Minning_Project/two_years_merged_and_reduced.csv')

WindSE1ML = Winddata[Winddata ['region'] == 'SE1']



WindSE1testML = WindSE1ML.copy()
WindSE1testML.drop(['time', 'cluster', 'region'],axis = 1, inplace=True)

# column_maxes = WindSE1test.max()
# df_max = column_maxes.max()
# column_mins = WindSE1test.min()
# df_min = column_mins.min()
# normalized_df = (WindSE1test - df_min) / (df_max - df_min)
for column in WindSE1testML.columns:
    WindSE1testML[column] = WindSE1testML[column]  / WindSE1testML[column].abs().max()
      
# view normalized data
#print(WindSE1test)

normalized_dfML = WindSE1testML








xds = normalized_dfML.iloc[:,:7].values
#print(x)
# xds = np.delete(xds,np.s_[0:2], axis=1)
#print(x)
yds= normalized_dfML.iloc[:,[8]].values
#print(y)





#Splitting the dataset into Training set and Test set
XML_train, XML_test, yML_train, yML_test = train_test_split(xds, yds, test_size = 0.2,random_state = 42)

#Fitting Simple Linear Regression to Traning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(XML_train,yML_train)


#Fitting Simple Linear Regression to Traning set

#Predicting the Test Set results
y_pred = regressor.predict(XML_test)
#print (regressor.coef_)
#print (regressor.intercept_)


# #Manual Building optimal model using backward Elimination
# '''import statsmodels.formula.api as sm
# x = np.append(arr = np.ones((61,1)).astype(float), values = x, axis =1)
# print(x)
# x_opt = x[:, [0,1,2,3]]
# x_opt = np.array(x_opt, dtype=float)
# print(x_opt)
# regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
# regressor_OLS.summary()'''

#Automatic Building optimal model using Backward Elimination 
XML_train = np.append(arr = np.ones((38476,1)).astype(float), values = XML_train, axis =1)
print (xds)
def backwardElimination(xtrain, sl):
    numVars = len(xds[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = yML_train, exog = XML_train).fit()
        print(regressor_OLS.summary())
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    xtrain = np.delete(xtrain, j, 1)
   
    
    return xtrain



 
SL = 0.05
x_opt = XML_train[:, [0, 1, 2, 3, 4, 5, 6]]
x_opt = np.array(x_opt, dtype=float)
X_Modeled = backwardElimination(x_opt, SL)
print(X_Modeled)
regressor_OLS = sm.OLS(endog = yML_train, exog = X_Modeled).fit()
x_testpredict  = np.delete(X_Modeled,np.s_[1:1], axis=1)
print(x_testpredict)
predvalues  = regressor_OLS.predict(x_testpredict)


df = pd.DataFrame({'Actual': yML_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

rms = sqrt(mean_squared_error(yML_train, regressor_OLS.predict(x_testpredict)))
print("RMSE:", rms)

# from sklearn.metrics import mean_squared_error
# from math import sqrt


# rms = sqrt(mean_squared_error(yML_train, regressor_OLS.predict(x_testpredict)))
# print("RMSE:", rms)
# plt.scatter(yML_train, regressor_OLS.predict(x_testpredict), color = 'red')
# plt.plot(XML_train, regressor_OLS.predict(x_testpredict), color = 'blue')

# plt.title ('Maximum Pressure vs VH (Test set)')
# plt.xlabel('Maximum Pressure')
# plt.ylabel('VH')
# plt.show()

#y_prednew = X_Modeled.predict(X_test)