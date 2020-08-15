#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:39:46 2020

@author: shreyamajumdar
"""
#Linear Regression using scikit library
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import FixData


#Collect X and Y
X = FixData.x #Take the x array from the Fix Data script
Y = FixData.y #Take the y array from the Fix Data script

#Specifies how much of the data should be used
#test_size is used to specify the test size, or train_size can be used
#random_state = 0 or leaving it blank sets it to use numpy's random generator
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)


#Create model
reg = LinearRegression()
#Training
reg = reg.fit(X, Y)
#Y predicted values
Y_pred = reg.predict(X)
#can also use this to find the predicted values for a single x, or for an array of values

#r^2 score calculation
r2_score = reg.score(X, Y)
print("The coefficient of determination, or the r squared coefficient: " + r2_score)


#Show the training set + regression line
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()