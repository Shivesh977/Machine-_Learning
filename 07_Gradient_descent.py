# Here we are applying simple gradient descent 

from sklearn.datasets import make_regression
import numpy as np  
import matplotlib.pyplot as plt 

X,y = make_regression(n_samples=4, n_features=1, n_informative=1, n_targets=1,noise=80,random_state=13) # picking a random sample which is almost linear 
# plt.scatter(X,y)
# plt.show()

#Applying ols technique to find m and b 

from sklearn.linear_model import LinearRegression 
reg=LinearRegression()
reg.fit(X,y)
slope = reg.coef_ # slope comes out to be 78.35
intercept = reg.intercept_ # intercept comes out to be 21.15 

# Now we will perform gradient descent approach to find b(intercept)
# let us take slope(m) = 78.35 
# let us assume starting value of b = 0 

m=78.35
b=0 

# calculating slope at b=0 that is inserting b=0 after differentiating loss function 


for i in range (1,10000):
    slope = -2 * np.sum(y-m*X.ravel()-b)
    lr = 0.1  # learning rate
    stepsize = lr*slope 
    b= b -stepsize
print("Intercept is : " ,b )


# Making a class gradient descent 
class GD_regression():
    def __init__(self,learning_rate,epoches): # constructor
        self.m=29.19
        self.b=-120
        self.lr=learning_rate
        self.epoches = epoches 
    def fit(self,X,y):
        for i in range (self.epoches):
            slope = -2 * np.sum(y-self.m*X.ravel()-self.b)
            self.b= self.b -(self.lr * slope)
        print("Intercept is : ",self.b)
        
gd=GD_regression(0.001,100000)
gd.fit(X,y)

        