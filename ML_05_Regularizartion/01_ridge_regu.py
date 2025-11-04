import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
data=load_diabetes()

X=data.data
y=data.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)

#  applying normal linear regression 
from sklearn.linear_model import LinearRegression
L=LinearRegression()
L.fit(X_train,y_train)

y_pred=L.predict(X_test)  # predicting the output on test data 

from sklearn.metrics import r2_score,mean_squared_error

print("R2 score",r2_score(y_test,y_pred)) # finding r2 score 
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
# R2 score 0.5188118914964637
# RMSE 48.72710829141399

# Now using ridge regularization 

from sklearn.linear_model import Ridge
R=Ridge(alpha=100000)
R.fit(X_train,y_train)

y_pred1=R.predict(X_test)

print("R2 score",r2_score(y_test,y_pred1)) # finding new r2 score after ridge regularization 
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred1)))
# R2 score 0.5188118914964637
# RMSE 48.72710829141399  almost same 


# analogy for various values of alpha how function changes 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_preds_ridge(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)

alphas = [0, 20, 200]
cs = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, 'b+', label='Datapoints')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    # Plot
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])], c, label='Alpha: {}'.format(alpha))

plt.legend()
plt.show()