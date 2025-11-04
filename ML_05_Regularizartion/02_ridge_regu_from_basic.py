class Ridge: # for 2Dimensional data 
    def __init__(self,alpha=0.1):
        self.alpha=alpha
        self.m=None
        self.b=None 
        
    def fit(self,X_train,y_train):
        num=0
        den=0  # denominator 
        for i in range(X_train.shape[0]):
            num+=(y_train[i]- y_train.mean())*(X_train[i]- X_train.mean())
            den+=(X_train[i]- X_train.mean())*(X_train[i]- X_train.mean()) 
        
        self.m=num/den + self.alpha 
        self.b=y_train.mean() - (self.m * X_train.mean())
        print(self.m,self.b)
        
    def predict(X_test):
        pass
    
    
# Ridge regularization for n-dimensional data 
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np

X,y = load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1,solver='cholesky')
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
r2_score(y_test,y_pred)

print(reg.coef_)
print(reg.intercept_)