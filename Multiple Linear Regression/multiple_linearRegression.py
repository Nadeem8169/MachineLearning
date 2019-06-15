#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:40:00 2019

@author: nadeem
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets
dataset=pd.read_csv('50_Startups.csv')
X=pd.DataFrame(dataset.iloc[:,:-1].values)
Y=dataset.iloc[:,4].values

#Encoding the profit column or categorical data in numbers
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X=LabelEncoder()
X.iloc[:, 3]=labelEncoder_X.fit_transform(X.iloc[:, 3])

#Avoiding the dummy variable trap
X=X[:,1:]

onehotEncoder=OneHotEncoder(categorical_features=[3])
X=onehotEncoder.fit_transform(X).toarray()

#spliting the datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predictin the test sets results
Y_pred=regressor.predict(X_test)