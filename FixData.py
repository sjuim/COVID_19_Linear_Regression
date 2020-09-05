#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:15:14 2020

@author: shreyamajumdar
"""
#This allows me to deal with the csv file
import pandas as pd
import numpy as np
import datetime

#Read the csv file note: this is working, it's accessing the right file
dataFrame = pd.read_csv('/Users/aparajitachatterjee/hack-workspace/covid-19-data/us-counties.csv', index_col = 'date') #when we start working with the actual dataset

#By doing this, this is not a new array holding the desired values from before, but rather an array of pointers to the desired values
desiredState = 'Massachusetts'
dataFrameState = dataFrame[dataFrame['state'] == desiredState]
desiredCounty = 'Middlesex'
dataFrameCounty = dataFrameState[dataFrameState['county'] == desiredCounty]
print(dataFrameCounty)

#Drop any empty columns
df = dataFrameCounty['cases'].dropna()

#Change the datetime format of the date column into numbers so that linear regression can be done
y = np.array(df.values, dtype = int)
x = np.array(pd.to_datetime(df).index.values, dtype = datetime.date)