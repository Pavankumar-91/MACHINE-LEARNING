# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# import dataset
dataset = pd.read_csv(r"C:\Users\pk161\OneDrive\DATA\Investment.csv")
dataset

# seperate the variable

# independent variable
x = dataset.iloc[:,:-1]

# dependent variable
y = dataset.iloc[:,4]

# converting categorical columns into binary columns
x = pd.get_dummies(x,dtype=int)

# split the data into training set and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# creating regression model
regressor = LinearRegression()

regressor.fit(x_train,y_train)

# predict the model
y_pred = regressor.predict(x_test)

# build mlr model
m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

x = np.append(arr = np.ones((50,1)).astype(int),values=x,axis=1)

# ordinary least squares
import statsmodels.api as sm

# selecting features
x_opt = x[:,[0,1,2,3,4,5]]

# ols regression
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
# detailed summary of the fitted OLS regression model
regressor_OLS.summary()



import statsmodels.api as sm

x_opt = x[:,[0,1,2,3,5]]

regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()

regressor_OLS.summary()


import statsmodels.api as sm

x_opt = x[:,[0,1,2,3]]

regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()

regressor_OLS.summary()


import statsmodels.api as sm

x_opt = x[:,[0,1,3]]

regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()

regressor_OLS.summary()


import statsmodels.api as sm

x_opt = x[:,[0,1]]

regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()

regressor_OLS.summary()


# evaluating the model performence

bias = regressor.score(x_train,y_train)
bias

variance = regressor.score(x_test,y_test)
variance