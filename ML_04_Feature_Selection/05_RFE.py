# We are implementing Recrusive Feature Elimination(RFE) 

################################## DOING MANUALLY
import pandas as pd
import numpy as np

df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv') # iris dataset 

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier() # using randomForestClassifier 

rf.fit(X,y)

rf.feature_importances_ # give feature_importance of each feature(column)

# in first iteration sepal width have minimum feature importance so we remove that
X.drop(columns='sepal_width',inplace=True)

# Again repeating above process until we get single best feature 
rf = RandomForestClassifier()
rf.fit(X,y)

rf.feature_importances_
X.drop(columns='sepal_length',inplace=True)

rf = RandomForestClassifier()
rf.fit(X,y)

rf.feature_importances_
X.drop(columns='petal_length',inplace=True)

################################ DOING ABOVE PROCESS USING SKLEARN LIBRARY 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load iris dataset
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
df = pd.read_csv(url)

# Separate features and target variable
X = df.drop("species", axis=1)
y = df["species"]

# Initialize RandomForestClassifier
model = RandomForestClassifier()

# Initialize RFE
rfe = RFE(estimator=model, n_features_to_select=1) # estimator asks for model which we are using and n_features give us no of features that we want 

# Fit RFE
rfe.fit(X, y)

# Print the ranking
ranking = rfe.ranking_
print("Feature ranking:")

for i, feature in enumerate(X.columns):
    print(f"{feature}: {ranking[i]}")

################################################ APPLYING RFECV (RFE cross validation) its better than rfecv 
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selector.support_
# array([ True,  True,  True,  True,  True, False, False, False, False,
    #    False])
selector.ranking_
# array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
