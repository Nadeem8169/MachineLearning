
"""
Created on Wed Jun 12 12:56:30 2019

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv('Data.csv')
X=pd.DataFrame(dataset.iloc[:, :-1])
y=pd.DataFrame(dataset.iloc[:,3])

#taking care of the missing data by using the Mean and median strategy
# we can use this strategy only for the numeric value
from  sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer=imputer.fit(X.iloc[:, 1:3])
x1=imputer.transform(X.iloc[:, 1:3])

#taking care of the missing data by using the constant  and most frequent strategy.
#we can use this strategy only for the numeric value and string value
imputer=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=6777,verbose=0)
imputer=imputer.fit(X.iloc[:, 1:3])
x2=imputer.transform(X.iloc[:, 1:3])






