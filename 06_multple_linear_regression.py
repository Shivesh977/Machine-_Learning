import numpy as np
from sklearn.datasets import load_diabetes # preloaded dataset already present in sklearn 

X,y = load_diabetes(return_X_y=True) # loading a dataset 
X,y # contains 442 rows  and 10 columns therefore 10 input columns 

# Using Sklearn's Linear Regression
from sklearn.model_selection import train_test_split # importing to divide dataset into train and test data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2) 

from sklearn.linear_model import LinearRegression #importing lr library from sklearn

reg = LinearRegression() # reg is object of linear regression 
reg.fit(X_train,y_train)  # training the model

y_pred = reg.predict(X_test) # predicting result 


reg.coef_ # gives the B0 
reg.intercept_ # give B1,B2,.... BN 