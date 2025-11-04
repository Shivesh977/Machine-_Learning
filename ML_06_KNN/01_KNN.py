# Applying knn on breast cancer prediction data

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df.drop(columns=['id','Unnamed: 32'],inplace=True) # as they are of no use

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0],test_size=0.2, random_state=2)

# Performing normalization to make all columns on same scale 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # applying z_score normalization 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) #n_neighbours is for choosing the neighbours no 

knn.fit(X_train,y_train) 

from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred) # accuracy is coming out to be 97% 


# Choosing values of k by experimentation 
y=[]
x=[]
for i in range (1,16):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,X_test)
    y_pred=knn.predict(X_test)
    accuracy_score(y_test,y_pred)
    y.append(accuracy_score)
    x.append(i)
    
    
plt.plot(x,y)
plt.show()
 
