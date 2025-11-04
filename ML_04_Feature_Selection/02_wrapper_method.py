# In this section we will perform wrapper method of feature selection 

from sklearn.datasets import load_iris # doing on iris_data set
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import cross_val_score

df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')



########################## 1 : Exhaustive Feature Selector 
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

lr = LogisticRegression() # linear regression 
 
sel = EFS(lr, max_features=4, scoring='accuracy', cv=5) # max_features means max size of subset that u wanna make 
# cv = 5 means we are calculating accuracy of particular combination 5 times to enhance reliability 

model = sel.fit(df.iloc[:,:4],df['species']) # training model  species here is output column 

(model.best_score_) # gives best score of one of combinations 
(model.best_feature_names_) # gives name of that columns 
(model.subsets_) # gives all possible combinations of columns and their score 

metric_df = pd.DataFrame.from_dict(model.get_metric_dict()).T # converting model.subsets_ into dataframe 

########################### 2 : Sequential backward selection / elimination :
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np 
import matplotlib.pyplot as plt 

# load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

# separate the target variable
X = data.drop("medv", axis=1)
y = data['medv']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

(X_train.shape)

# scaling the dataframe to normalize or make to a same level 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LinearRegression() # regression model 

# Base Model is created in which initially all input / features col are taken and score is calculated 
print("training",np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='r2')))
print("testing",np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='r2')))


lr = LinearRegression() 

# perform backward elimination
sfs = SFS(lr, k_features='best', forward=False, floating=False, scoring='r2',cv=5) # lr means model which u are sending , k_features means  kitne features chahiye . .if no preference give best 
# forward = false means we have to apply backward elimination 

sfs.fit(X_train, y_train) # train the model 

sfs.k_feature_idx_ # gives indices of best columns 

metric_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T

metric_df['observations'] = 404
metric_df['num_features'] = metric_df['feature_idx'].apply(lambda x:len(x))
metric_df['adjusted_r2'] = adjust_r2(metric_df['avg_score'],metric_df['observations'],metric_df['num_features'])

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# #  plotting the graph 
# fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_err',)

# plt.title('Sequential Backward Elimination (w. StdErr)')
# plt.grid()
# plt.show()



X_train_sel = sfs.transform(X_train)
X_test_sel = sfs.transform(X_test)

model = LinearRegression()

print("training",np.mean(cross_val_score(model, X_train_sel, y_train, cv=5, scoring='r2')))
print("testing",np.mean(cross_val_score(model, X_test_sel, y_test, cv=5, scoring='r2'))) # gives new score after taking that best columns 



################################### 3 : Forward Selection method

# no changes in code  when compared to backward ones 
# only changes is :  make forward = True for forward selection 
sfs = SFS(lr, k_features='best', forward=True, floating=False, scoring='r2',cv=5)



###################### USING SK_LEARN 

from sklearn.feature_selection import SequentialFeatureSelector as SFS

sfs2 = SFS(model,
           n_features_to_select=5,
           direction='forward',
           scoring='r2',
           n_jobs=-1,
           cv=5)

sfs2 = sfs2.fit(X_train, y_train)
np.arange(X.shape[1])[sfs2.support_]