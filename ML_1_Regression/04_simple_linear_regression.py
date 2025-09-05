# Here we are importing a data set representing cgpa and package of students 
# We have to develop a simple linear regression model to predict package of student based on given cgpa 

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt   
df=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Datafile_csv\placement.csv")
# print(df) 

# plotting scatter plot 
x=df['cgpa']
y=df['package']
plt.scatter(x,y)
# plt.show()

# Steps of linear regression 
# seperate input and output 
x=df.iloc[:,0:1] # all rows col=1 i.e only cgpa column  : input 
y=df.iloc[:,-1] # all rows col = last i.e only package  : output 

# Dividing data into  for training and testing 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2) # test_size = 0.2 means take 20 percent data for test and 80 for training random_state =2 means so that we can reproduce result 

from sklearn.linear_model import LinearRegression 
lr=LinearRegression() # making object 
lr.fit(x_train,y_train) # function of fit method is to train the data 

# predicting the result based on x_test , y_test 
result = lr.predict(x_test.iloc[0].values.reshape(1,1)) 
# print(result)

# m and b of line 
m=lr.coef_
b=lr.intercept_
# now we can calculate y : package based on any given cgpa as y=mx + b 
