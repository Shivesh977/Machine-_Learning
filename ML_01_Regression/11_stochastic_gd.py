
from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time

X,y = load_diabetes(return_X_y=True)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
# reg = LinearRegression()
# reg.fit(X_train,y_train)
# LinearRegression()
# print(reg.coef_)
# print(reg.intercept_)

# y_pred = reg.predict(X_test)
# r2_score(y_test,y_pred)

####################################################################################################################################################
# Now making our own class 

class GDRegressor:
    def __init__(self,learning_rate,epochs):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
        self.intercept_=0
        self.coef_=np.ones(X_train.shape[1]) # n no of coeffecients B0,b1,b2 ets all initallised to 1 
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]) : # will run till no of rows 

                idx =np.random.randint(0,X_train.shape[0])  # select random row 
                
                # prediction for that particular index 
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_ # y_hat means Y_predicted  here now we will get a single no.
                intercept_der = -2 * (y_train[idx] - y_hat) # we are not doing summation because we are doing it for single row not for whole data 
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                # Predicting coeffecient 
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx]) 
                self.coef_ = self.coef_ - (self.lr * coef_der)
                
        print("Intercept is : ",self.intercept_)
        print("Coeffeceint is : ",self.coef_) 
                
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_


# sgd=GDRegressor(0.01,50)
# sgd.fit(X_train,y_train)
# y_pred=sgd.predict(X_test)
# print(r2_score(y_test,y_pred))

####################################################################################################################################################
# Using builting stochastic gradient descent of sklearn 
from sklearn.linear_model import SGDRegressor 

reg=SGDRegressor(max_iter=100,learning_rate='constant',eta0=0.01) # max_iter is epoch 

reg.fit(X_train,y_train)
y_predict = reg.predict(X_test)
r2_score(y_test,y_predict)
