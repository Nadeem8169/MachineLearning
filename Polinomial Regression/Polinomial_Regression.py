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
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1].values
X=X.reshape(-1,1)
Y=dataset.iloc[:,2].values

#fitting Linear REgression to the dataset
from sklearn.linear_model import LinearRegression
Lin_reg=LinearRegression()
Lin_reg.fit(X,Y)

#Visualizing the Linear Linear
plt.scatter(X,Y,color="red")
plt.plot(X,Lin_reg.predict(X),color='blue')
plt.title("Sales vs Level(Linear Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


#fitting Polinomial REgression to the dataset
from sklearn.preprocessing import PolynomialFeatures
Poly_reg=PolynomialFeatures(degree=4)
X_poly=Poly_reg.fit_transform(X)
lin_reg1=LinearRegression()
lin_reg1.fit(X_poly,Y)

#Visualizing Ploinomial Regression Model
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg1.predict(Poly_reg.fit_transform(X)),color='blue')
plt.title("Sales vs Level(Polynomial Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Predicting the results using the Linear Regression
Lin_reg.predict([[6.5]])



#Predicting the results using the Linear Regression
lin_reg1.predict(Poly_reg.fit_transform([[6.5]]))





"""
Created on Sat Jun 15 20:39:31 2019

@author: nadeem
"""

