#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:12:28 2020

@author: shreyamajumdar
"""
#Hard coding the linear regression without machine learning libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#Read data
data = pd.read_csv() #name of dataset goes in the parenthesis

#This gives an output of the shape of the array, in this format: (rows, columns)
print(data.shape)

#Outputs the first few columns in the data set
data.head()

#Collect X and Y
X = data[].values #Inside the [] will go the header of the X values (whatever is the header of the X values column will go in there as a string)
Y = data[].values #Inside the [] will go the header of the Y values (whatever is the header of the Y values column will go in there as a string)

#Calculate the regression coefficients using formulas
#Means of the x and y values
mean_x = np.mean(X)
mean_y = np.mean(Y)

#Number of data points
n = len(X)

#Formula calculations
numer = 0
denom = 0
for i in range(n):
    numer = numer + (X[i] - mean_x) * (Y[i] - mean_y)
    denom = denom + (x[i] - mean_x)**2
a = numer/denom
b = mean_y - (a * mean_x)

#Print coefficients
print(a, b)


#Plot the graph with the regression line
max_x = np.max(X)+100
min_x = np.min(X)-100

#Calculate the values for the axes
#This generates the values for the x axis, by giving the start, stop, and number of evenly spaced numbers in this interva;
x = np.linspace(min_x, max_x, 1000)
#Generates the appropriate values for the y axis according to the regression line in order to give the correct scale of the graph
y =  a*x + b

#Plot the line
plt.plot(x, y, color = '#58b970', label = 'Regression Line')
#Plot the data points
plt.scatter(X, Y, color = '#ef5423', label = 'Scatterplot')

#Labels
plt.xlabel() #Label of the x axis goes in the parenthesis
plt.ylabel() #Label of the y axis goes in the parenthesis
plt.legend()
plt.show()



#Calculate the r^2 value to determine how good of a fit the line is
ss_t = 0
ss_r = 0
for i in range(n):
    ss_t = ss_t + (Y[i] - mean_y)**2 #Calculates the square of the difference from the mean
    y_pred = (a * X[i]) + b
    ss_r = (Y[i] - y_pred)**2 #Calculates the square of the residuals
r2 = 1 - (ss_r/ss_t)
print(r2)









#Linear Regression using scikit library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
import numpy as np
import matplotlib.pyplot as plt


#Read data
data = pd.read_csv() #name of dataset goes in the parenthesis

#Collect X and Y
X = data[].values #Inside the [] will go the header of the X values (whatever is the header of the X values column will go in there as a string)
Y = data[].values #Inside the [] will go the header of the Y values (whatever is the header of the Y values column will go in there as a string)

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
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

