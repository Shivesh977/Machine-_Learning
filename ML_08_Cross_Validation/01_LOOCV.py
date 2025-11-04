

# POINT TO BE NOTED IS THAT CV IS USED FOR MODEL EVALUTATION NOT FOR MODEL SELECTION 



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import pandas as pd 

# Load the Boston Housing dataset
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

# Create a linear regression model
model = LinearRegression()

# Create a LeaveOneOut cross-validator
loo = LeaveOneOut()

# Use cross_val_score for the dataset with the model and LOOCV
# This will return the scores for each iteration of LOOCV
scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error') # cv=loo stands for we have to use leave one out method

mse_scores = -scores  # Invert the sign of the scores

# Print the mean MSE over all LOOCV iterations
print("Mean MSE:", mse_scores.mean())