#Here in this we will take a dataset consisting of 563 columns 
# we will apply logistic regression on this then we will calculate its accuracy lets x 

# After that , we wil apply feature selection to reduce no of columns to 100
# we will apply feature selection to reduce  no of columns to 100
# again we will apply logistic regression ..then we will calculate its accuracy lets y 

# In end we will show y is close to x ..therefore after feature selection our model prediction or performace doesnt decrease much

import numpy as np     
import pandas as pd 

df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Datafile_csv\train.csv").drop(columns='subject')

######################################################## CALCULATING X ##########################################################################################

from sklearn.preprocessing import LabelEncoder # to convert strings into no such that ML model can be run on that 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separate features and target
X = df.drop('Activity', axis=1)
y = df['Activity']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############### Applying logistic regression  
# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=1000)  # Increase max_iter if it doesn't converge
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Calculate and print accuracy score
accuracy = accuracy_score(y_test, y_pred)
# print("Test accuracy:", accuracy) # Test accuracy or   x is 98 

# Therefore X is 98

################################################################## FEATURE SELECTION #######################################################

######################## 1 REMOVING DUPLICATE COLUMNS 


# Remove duplicate features
# Get the subset of columns with duplicate values
duplicated_cols = df.columns[df.T.duplicated()]

# Remove the duplicated columns
data = df.drop(columns=duplicated_cols)


# other method 

def get_duplicate_columns(df):
    duplicate_columns = {}
    seen_columns = {}

    for column in df.columns:
        current_column = df[column]

        # Convert column data to bytes
        try:
            current_column_hash = current_column.values.tobytes()
        except AttributeError:
            current_column_hash = current_column.to_string().encode()

        if current_column_hash in seen_columns:
            if seen_columns[current_column_hash] in duplicate_columns:
                duplicate_columns[seen_columns[current_column_hash]].append(column)
            else:
                duplicate_columns[seen_columns[current_column_hash]] = [column]
        else:
            seen_columns[current_column_hash] = column

    return duplicate_columns

duplicate_columns = get_duplicate_columns(X_train) # this give list of duplicate columns in form of dictionary where key is original col and values are duplicate columns 

for one_list in duplicate_columns.values(): # removing duplicate columns 
    X_train.drop(columns=one_list,inplace=True)
    X_test.drop(columns=one_list,inplace=True)

# we have reduced 20 duplicate columns 

############################# 2 : APPLYING VARIENCE THRESHOLD 

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.05)

sel.fit(X_train) # find the columns whose var is less than threshold 

columns = X_train.columns[sel.get_support()] # gives columns whose var is greater than threshold val 

X_train = sel.transform(X_train) # deleting those col where var < threshold  and dataframe is converted into numpy array 
X_test = sel.transform(X_test)

X_train = pd.DataFrame(X_train, columns=columns)  # converting into dataframe again
X_test = pd.DataFrame(X_test, columns=columns)

# Now we have 349 columns left 

########################### 3 : Correlation 

corr_matrix = X_train.corr()
# Get the column names of the DataFrame
columns = corr_matrix.columns

# Create an empty list to keep track of columns to drop
columns_to_drop = []

# Loop over the columns
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        # Access the cell of the DataFrame
        if corr_matrix.loc[columns[i], columns[j]] > 0.95:
            columns_to_drop.append(columns[j])

columns_to_drop = set(columns_to_drop)

X_train.drop(columns = columns_to_drop, axis = 1, inplace=True)
X_test.drop(columns = columns_to_drop, axis = 1, inplace=True)

# we have dropped 197 more columns so now we have 152 columns 


############################# 4 : ANOVA 
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


sel = SelectKBest(f_classif, k=100).fit(X_train, y_train) # give k best columns 

# display selected feature names
X_train.columns[sel.get_support()]

columns = X_train.columns[sel.get_support()]
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)
print(X_train.shape)
print(X_test.shape)

# Now we have 100 columns left 

################################################################# Calculating Y ##################################################################
# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=1000)  # Increase max_iter if it doesn't converge
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Calculate and print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy) # test accuracy comes out to be 97%



################################################################# CHI SQUARE TEST ################################################################

#ONLY APPLICABLE IF BOTH INPUT AND OUTPUT COLUMNS ARE CATEGORICAL 
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')[['Pclass','Sex','SibSp','Parch','Embarked','Survived']]
# titanic.head()
ct = pd.crosstab(titanic['Survived'],titanic['Sex'],margins=True)

from scipy.stats import chi2_contingency
chi2_contingency(ct)

score = []

for feature in titanic.columns[:-1]:
    
    # create contingency table
    ct = pd.crosstab(titanic['Survived'], titanic[feature])
    
    # chi_test
    p_value = chi2_contingency(ct)[1]
    score.append(p_value)

pd.Series(score, index=titanic.columns[:-1]).sort_values(ascending=True).plot(kind='bar')


# another method to do above procedure 

# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_selection import chi2
# import matplotlib.pyplot as plt

# # assuming titanic is your DataFrame and 'Survived' is the target column

# # Encode categorical variables
# le = LabelEncoder()
# titanic_encoded = titanic.apply(le.fit_transform)

# X = titanic_encoded.drop('Survived', axis=1)
# y = titanic_encoded['Survived']

# # Calculate chi-squared stats
# chi_scores = chi2(X, y)

# # chi_scores[1] are the p-values of each feature.
# p_values = pd.Series(chi_scores[1], index = X.columns)
# p_values.sort_values(inplace = True)

# # Plotting the p-values
# p_values.plot.bar()

# plt.title('Chi-square test - P-values')
# plt.xlabel('Feature')
# plt.ylabel('P-value')

# plt.show()