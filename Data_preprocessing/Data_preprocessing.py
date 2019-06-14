"""
Created on Wed Jun 12 15:43:46 2019
@author: nadeem
"""

#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import the dataset
dataset=pd.read_csv('Data.csv')
X=pd.DataFrame(dataset.iloc[:, :-1].values)
Y=pd.DataFrame(dataset.iloc[:, 3].values)


#Taking care the missing values
from  sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer=imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3]=imputer.transform(X.iloc[:, 1:3])




#categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelEncoder=LabelEncoder()
X.iloc[:, 0]=labelEncoder.fit_transform(X.iloc[:, 0])
Ylabel=labelEncoder.fit_transform(Y.iloc[:, 0])

onehotEncoder=OneHotEncoder(categorical_features=[0])
X=onehotEncoder.fit_transform(X).toarray()

#spliting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=0)



#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)










