# Now here we are taking both m and b as unknown 


from sklearn.datasets import make_regression
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
x,y = make_regression(n_samples=100,n_features=1, n_informative=1,n_targets=1,noise=20,random_state=13) # picking a random sample which is almost linear 

class GD_regression:
    def __init__(self,m,b,learning_rate,epoches):
        self.m=m
        self.b=b
        self.lr=learning_rate
        self.epoches=epoches 
    def fit(self,x,y):
        for i in range(self.epoches):
            intercept = -2 * np.sum(y-self.m*x.ravel()-self.b)
            self.b=self.b-intercept*self.lr 
            slope =  -2 * np.sum((y-self.m*x.ravel()-self.b)*x.ravel())
            self.m=self.m-slope*self.lr 
        print("slope is : ",self.b)
        print("Intercept is : ", self.m)
    def predict(self,x):
        print(self.m*x + self.b)


gd=GD_regression(100,-120,0.001,100)
gd.fit(x,y)
gd.predict(27)