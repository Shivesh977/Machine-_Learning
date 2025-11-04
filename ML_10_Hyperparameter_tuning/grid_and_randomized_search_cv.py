# We try all possible combination of hyperparameters and find the one which best suits 
# Brute force approach 


import numpy as np
import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv') # importing boston housing dataset 
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import cross_val_score,KFold
from sklearn.neighbors import KNeighborsRegressor
# First using default  given in libraries

knn = KNeighborsRegressor()
kfold = KFold(n_splits=5, shuffle=True, random_state=1) # here parameters choosen are default one 
scores = cross_val_score(knn, X, y, cv=kfold, scoring='r2')
scores.mean()  # 0.4761976351913221  we have to improve this using gridsearch cv 

############################### Applying GridSearchCV  to find best combination of hyperparameters #############################
from sklearn.model_selection import GridSearchCV   
knn = KNeighborsRegressor()

param_grid = {
    'n_neighbors':[1,3,5,7,10,12,15,17,20], # taking all these values for n_neighbours 
    'weights':['uniform','distance'],
    'algorithm':['ball_tree', 'kd_tree', 'brute'],
    'p':[1,2]
} # specifying the parameter space 

gcv = GridSearchCV(knn, param_grid, scoring='r2', refit=True, cv=kfold, verbose=2) # object of gridsearch cv  model used is knn, param_grid,scoring used is r2 score , crossvalidation(cv)= kfold .. verbose : prints resuls at each iteration (combination) of parameters 

gcv.fit(X,y) # train the model 

gcv.best_params_ # gives combination of parameters which gives best results 

gcv.best_score_ # prints the best score that we can get using best_params  0.6117139367845081 score is improved from 0.4 to 0.6 r2 score 

gcv.cv_results_ # gives combination results in form of dictionary 

pd.DataFrame(gcv.cv_results_)[['param_algorithm',	'param_n_neighbors',	'param_p', 'param_weights', 'mean_test_score']].sort_values('mean_test_score',ascending=False) # printing that as a dataframe in the form of table 



########################################### Applying randomized search cv on same data ######################################################

from sklearn.model_selection import RandomizedSearchCV

rcv = RandomizedSearchCV(knn, param_grid, scoring='r2', refit=True, cv=kfold, verbose=2) # object ... 

rcv.fit(X,y) 

rcv.best_score_ # gives best score  0.589758956010885 but score coming by gridsearch cv was more better  0.61

rcv.best_params_ # gives combination of params which gives best result .... 0.58

