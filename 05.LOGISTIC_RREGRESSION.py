# import libraries
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Read the dataset
dataset = pd.read_csv(r"C:\Users\pk161\OneDrive\DATA\logit classification.csv")


# independent variable
x = dataset.iloc[:,[2,3]].values

# dependent variable
y = dataset.iloc[:,-1].values


# Split the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# Feature Scalling
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



# Logistic Regression Model

classifier = LogisticRegression()

# train the model using training data
classifier.fit(x_train,y_train)

# Predict the Logistic Regression Model
y_pred = classifier.predict(x_test)



# Confusion Matrix For Evaluating Model Performence
cm = confusion_matrix(y_test,y_pred)

print(cm)



# Accuracy Score
ac = accuracy_score(y_test,y_pred)

print(ac)



# Classification Report
cr = classification_report(y_test,y_pred)

print(cr)



# bias score
bias = classifier.score(x_train,y_train)

print(bias)



# variance score
variance = classifier.score(x_test,y_test)

print(variance)



# FUTURE PREDICTION

dataset1 = pd.read_csv(r"C:\Users\pk161\OneDrive\DATA\Future prediction1.csv")

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[2,3]].values

# Feature Scaling
sc = StandardScaler()

scaled_features = sc.fit_transform(dataset1)


# Creating an Empty DataFrame
y_pred1 = pd.DataFrame()

d2["y_pred1"] = classifier.predict(scaled_features)


d2.to_csv('pred_model.csv')


# To get path
import os
os.getcwd()
