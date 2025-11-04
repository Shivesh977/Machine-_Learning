import numpy as np 
import pandas as pd 

from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import ElasticNet 

df=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Datafile_csv\HousingData.csv")
# print(df)

X = df.drop("MEDV", axis=1)   # drop MEDV from features
y = df["MEDV"]                # target column


# Drop rows with NaN
X = X.dropna()
y = y[X.index]  # Keep y aligned


# importing train test split 
from sklearn.model_selection import train_test_split # importing to divide dataset into train and test data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)  


# Linear regression 
lr =LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
print("R2 score from linear regression ",r2_score(y_test,y_pred)) # 0.7605719989452936

# Ridge regression 
ridge = Ridge(alpha = 0.1)
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
r2_by_ridge= r2_score(y_test,y_pred) # 0.7627798381751651
print("R2_score by ridge regression : ",r2_by_ridge)

# Lasso regression 
lasso =Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
y_pred=lasso.predict(X_test)
r2_by_lasso=r2_score(y_test,y_pred) # 0.7627798381751651
print("R2 score by lasso regression : ", r2_by_ridge)

# Elastic net regression 
elastic =ElasticNet(alpha=0.1)
elastic.fit(X_train,y_train)
y_pred=elastic.predict(X_test)
r2_by_elastic=r2_score(y_test,y_pred) # 0.7986008749828191
print("R2 score by elastic regression : ", r2_by_elastic) 

# Therefore best result is given by Elastic net regression method


