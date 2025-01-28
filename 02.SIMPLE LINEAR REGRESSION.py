# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# reading dataset

dataset = pd.read_csv(r"C:\Users\pk161\VS_CODE\PYTHON_PROJECTS\MACHINE_LEARNING\Salary_Data.csv")
print(dataset)

# seperating the dataset into indipendent variable & dipendent variable

# indipendent variable
x = dataset.iloc[:,:-1]
print(x)

# dependent variable
y = dataset.iloc[:,1]
print(y)


# split the dataset into training set & testing set

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# reshaping the data
x_train = x_train.values.reshape(-1,1)

x_test = x_test.values.reshape(-1,1)

# creating the linear regression model

regressor = LinearRegression()
# fitting the model
regressor.fit(x_train,y_train)

# prediction
y_pred = regressor.predict(x_test)


# visualize the training set

plt.scatter(x_train,y_train,color='red')

plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.title("SALARY VS EXPERIENCE(TRAINING SET)")

plt.xlabel("YEARS OF EXPERIENCE")

plt.ylabel("SALARY")

plt.show()


# visualize the testing set

plt.scatter(x_test,y_test,color='green')

plt.plot(x_train,regressor.predict(x_train),color='yellow')

plt.title("SALARY VS EXPERIENCE(TESTING SET)")

plt.xlabel("YEARS OF EXPERIENCE")

plt.ylabel("salary")

plt.show()



# compare actual vs prediction

comparision = pd.DataFrame({'Actual' : y_test,'Predicted' : y_pred })

print(comparision)


# predict the salary for the 12 & 20 years

y_12 = regressor.predict([[12]])

y_20 = regressor.predict([[20]])

print(f"Predicted Salary For 12 Years Of Experience : ${y_12[0]:,.2f}")

print(f"Predicted Salary For 20 Years Of Experience : ${y_20[0]:,.2f}")



# evaluating the model performence

bias = regressor.score(x_train,y_train)

variance = regressor.score(x_test,y_test)


train_mse = mean_squared_error(y_train,regressor.predict(x_train))

test_mse = mean_squared_error(y_test,y_pred)


print(f"Training Score (R^2) : {bias:.2f}")

print(f"Testing Score (R^2) : {variance:.2f}")

print(f"Train MSE : {train_mse:.2f}")

print(f"Test MSE :{test_mse:.2f}")


# save the trained model

filename = "Simple_Linear_Regression_Model.pkl"

with open(filename,'wb') as file:
    pickle.dump(regressor,file)

print("Model has been pickled and saved as Simple_Linear_Regression_model.pkl")