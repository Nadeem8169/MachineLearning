"""
Created on Fri Jun 14 16:54:03 2019

@author: nadeem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,0].values
X=X.reshape(-1,1)
Y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#fitting Simple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predict the Test set result
Y_pred=regressor.predict(X_test)

#visualize the training set results
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Year of Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

#visualize the Test set results
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs Year of Experience(Test set)' )
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()