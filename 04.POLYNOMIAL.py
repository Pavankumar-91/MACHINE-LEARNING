# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# importing dataset
dataset = pd.read_csv(r"C:\Users\pk161\VS_CODE\PYTHON_PROJECTS\MACHINE_LEARNING\emp_sal.csv")

# indipendent variable
x = dataset.iloc[:,1:2].values

# dependent variable
y = dataset.iloc[:,2].values



# LINEAR REGRESSION MODEL
lin_reg = LinearRegression()

lin_reg.fit(x,y)

# LINEAR REGRESSION VISUALIZATION
plt.scatter(x,y,color='red')

plt.plot(x,lin_reg.predict(x),color='blue')

plt.title('Linear Regression Model(Linear Regression)')

plt.xlabel('POSOTION LEVEL')

plt.ylabel('SALARY')

plt.show()

# PREDICTION USING LINEAR REGRESSION 
lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)




# POLYNOMIAL REGRESSION MODEL
poly_reg = PolynomialFeatures(degree=5)

x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(x_poly,y)

# POLYNOMIAL REGRESSION VISUALIZATION
plt.scatter(x,y,color='green')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'yellow')
plt.title("POLYNOMIAL REGRESSION MODEL")
plt.xlabel('POSITION LEVEL')
plt.ylabel('SALARY')
plt.show()

# PREDICTION USING POLYNOMIAL REGRESSION MODEL
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)




# SVR  REGRESSION MODEL
svr_regressor = SVR(kernel='poly',degree=4,gamma='auto')

svr_regressor.fit(x,y)

# SVR MODEL VISUALIZATION

plt.scatter(x,y,color='blue')

plt.plot(x,svr_regressor.predict(x),color='red')

plt.title("SUPPORT VECTOR REGRESSION MODEL")

plt.xlabel('POSITION LEVEL')

plt.ylabel('SALARY')

plt.show()

# PREDICTION USING SVR MODEL
svr_model_pred = svr_regressor.predict([[6.5]])

print(svr_model_pred)




# KNN REGRESSION MODEL
knn_reg_model = KNeighborsRegressor(n_neighbors=5,weights='distance',p=2)

knn_reg_model.fit(x,y)

# KNN MODEL VISUALIZATION
plt.scatter(x,y,color='blue')

plt.plot(x,knn_reg_model.predict(x),color='red')

plt.title("K-NEAREST NEIGHBORS REGRESSION MODEL")

plt.xlabel('POSITION LEVEL')

plt.ylabel('SALARY')

plt.show()

# PREDICTION USING KNN MODEL
knn_reg_pred = knn_reg_model.predict([[6.5]])

print(knn_reg_pred)




# DECISION TREE REGRESSION MODEL
dt_reg = DecisionTreeRegressor()

dt_reg.fit(x,y)

# DECISION TREE VISUALIZATION
plt.scatter(x,y,color='blue')

plt.plot(x,dt_reg.predict(x),color='red')

plt.title(" DECISION TREE REGRESSION MODEL")

plt.xlabel('POSITION LEVEL')

plt.ylabel('SALARY')

plt.show()

# PREDICTION USING DECISION TREE REGRESSION MODEL
dt_reg_pred = dt_reg.predict([[6.5]])

print(dt_reg_pred)




# RANDOM FOREST REGRESSION MODEL
rf_reg = RandomForestRegressor(n_estimators=15)

rf_reg.fit(x,y)

# RANDOM FOREST REGRESSION MODEL VISUALIZATION

plt.scatter(x,y,color='blue')

plt.plot(x,rf_reg.predict(x),color='red')

plt.title('RANDOM FOREST REGRESSION MODEL')

plt.xlabel('POSITION LEVEL')

plt.ylabel('SALARY')

plt.show()

# PREDICTION USING RANDOM FOREST REGRESSION MODEL
rf_reg_pred = rf_reg.predict([[6.5]])

print(rf_reg_pred)
