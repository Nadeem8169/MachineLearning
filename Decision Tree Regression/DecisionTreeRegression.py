
"""
Created on Sun Jun 16 11:46:43 2019

@author: nadeem
"""

import pandas as pd￼
import numpy as np
import matplotlib.pyplot as plt

#import the datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#fitting the Descision Tree Rgression to the datasets
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

#Visulising the Decision Tree Regression
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Truth or Bluff( Decision Tree Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#preficting the results
regressor.predict([[6.5]])