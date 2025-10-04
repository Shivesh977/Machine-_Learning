import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split 

# normal linear regression 

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

plt.scatter(X,y)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train,y_train)
print(reg.coef_)  # [26.43941934] 
print(reg.intercept_) # -2.5331087147574536

# applying lasso regression 
alpha = [0,1,5,10,30]
plt.figure(figsize=(12,6))
plt.scatter(X,y)
for i in alpha:
    L=Lasso(alpha=i)
    L.fit(X_train,y_train)
    plt.plot(X_test,L.predict(X_test),label='alpha={}'.format(i))
    
plt.legend()
plt.show()# we can see in lasso as alpha(lambda) increases the wheitage of coeffeceint decreases and at alpha = 30 coeffecient shrinks to 0 
# ... but in ridge reg it tends to 0 not exactly 0 
