#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:15:14 2020

@author: aparajitachatterjee
"""
#This allows me to deal with the csv file

import pandas as pd
#Read the csv file
dataFrame = pd.read_csv('sampleData.csv', index_col = 'Name')



#this section isn't working, it's meant to drop all rows that have 'F' in the gender column
#How to delete rows from the csv file
for row in dataFrame.itertuples():
    print(row.index)  #each variable row is actually a series I think
    print(dataFrame.at[row, 'Gender'])
        #axis = 0 means delete rows, 1 means columns, #inplace =  false means do not edit the orginal data set
    dataFrame.drop(row)


#Add a new column
dataFrame['Days After'] = 0

day = 0
#How to change the value of the new column
for row in dataFrame.itertuples():
    dataFrame.loc[day:day+1, 'Days After'] = day
    day+=1
print(dataFrame)

