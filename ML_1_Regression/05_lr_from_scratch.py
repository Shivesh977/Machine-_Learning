import numpy as np    
import pandas as pd 


class LR :
    def __init__(self):# constructor 
        self.m=None
        self.b=None
    def fit(self,x_train,y_train): # training the model
        numerator=0
        denominator=0
        x_mean=x_train.mean()
        y_mean=y_train.mean()
        for i in range (x_train.shape[0]): # calculating m(slope)
            numerator+=(x_train[i]-x_mean)*(y_train[i]-y_mean)
            denominator+=(x_train[i]-x_mean)**2
            
        self.m=numerator/denominator # slope of line 
        self.b=y_mean-(self.m*x_mean )
        
    def predict(self,x_test):
        return (self.m * x_test) + self.b # return prediction 
         
         
df=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Datafile_csv\placement.csv")
x=df.iloc[:,0].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

lr=LR()
lr.fit(x_train,y_train)
print("Package of student on basis of given cgpa : ",lr.predict(x_test[0]))
