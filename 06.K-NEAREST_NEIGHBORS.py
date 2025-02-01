# Import Libraries
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

# Read the Dataset
dataset = pd.read_csv(r"C:\Users\pk161\OneDrive\DATA\logit classification.csv")


# independent variable
x = dataset.iloc[:,[2,3]].values

# dependent variable
y = dataset.iloc[:,-1].values


# Split the Dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



# FEATURE SCALING
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)




# K-NEAREST NEIGHBORS MODEL
classifier = KNeighborsClassifier(n_neighbors=4,p=2)

classifier.fit(x_train,y_train)

# PREDICT USING KNN MODEL
y_pred = classifier.predict(x_test)



# CONFUSION MATRIX
cm = confusion_matrix(y_test,y_pred)

print(cm)



# ACCURACY SCORE OF KNN-MODEL
ac = accuracy_score(y_test,y_pred)

print(ac)



# CLASSIFICATION REPORT OF KNN-MODEL
cr = classification_report(y_test,y_pred)

print(cr)


# BIAS 
bias = classifier.score(x_train,y_train)

print(bias)


# VARIANCE
variance = classifier.score(x_test,y_test)

print(variance)