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

onehotEncoder=OneHotEncoder(categorical_features=[3])
X=onehotEncoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X=X[:,1:]


#spliting the datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predictin the test sets results
Y_pred=regressor.predict(X_test)

#########----------------------------#######################

#Building the Optimal model using the Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#taking the optimal variables
X_opt=X[:,[0,1,2,3,4,5]]

#step 0:-significance level=0.05
#step1:-fit the model with all possible predictor
regressor_OLS=sm.OLS(endog=Y,  exog= X_opt).fit()

#step2:consider the predictor with highest P-values.if P>SL then goto step 3 otherwise finish
regressor_OLS.summary()

#Step 3:-remove the predictor which P-value >SL and fit the all possible model
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,  exog= X_opt).fit()
#Continuing the process till the P<SL
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,  exog= X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,  exog= X_opt).fit()
#Continuing the process till the P<SL
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

















