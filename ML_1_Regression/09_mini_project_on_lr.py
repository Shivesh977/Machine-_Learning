from sklearn.datasets import make_regression
import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LinearRegression  

df = pd.read_excel(r"students_dataset_2000.xlsx") 

x = df[['cgpa']].values     # 2D numpy array
y = df['package(LPA)'].values   # 1D numpy array

#Using gradient descent method 

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
 
        
gd=GD_regression(100,-120,0.001,1000)
gd.fit(x,y)
# gd.predict()

            
        
        
        
    