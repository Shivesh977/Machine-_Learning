import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

# We have a dataset which includes whether the given sentence has a positive or negative sentiment
# Given new dataset we have to predict whether sentiment is positive or negative 
# Using naive bayes .... using bayes probability 


######################################3######## Text preprocesssing #######################################################
# Text Cleaning 
# Sample 10000 rows
# Remove html tags
# Remove special characters
# Converting every thing to lower case
# Removing Stop words
# Stemming

df['sentiment'].replace({'positive':1,'negative':0},inplace=True) # replacing positive with 1 and neg with 0 

import re # importing regex 
clean = re.compile('<.*?>')
re.sub(clean, '', df.iloc[2].review)

# Function to clean html tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

df['review']=df['review'].apply(clean_html)

# converting everything to lower
def convert_lower(text):
    return text.lower()

df['review']=df['review'].apply(convert_lower) 

# function to remove special characters
def remove_special(text):
    x=''
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x + ' '
    return x

df['review']=df['review'].apply(remove_special)

# Remove the stop words
import nltk
from nltk.corpus import stopwords
stopwords.words('english')

def remove_stopwords(text):
    x=[]
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y
df['review']=df['review'].apply(remove_stopwords)


# Perform stemming
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
y=[]
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z
        
df['review']=df['review'].apply(stem_words)

# Join back   as data was converted into list joining back and making it string 
def join_back(list_input):
    return " ".join(list_input)
df['review']=df['review'].apply(join_back)


################################################################## Text vectorization ########################################################
X=df.iloc[:,0:1].values

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)   # take top 2500 features 

X=cv.fit_transform(df['review']).toarray() # text vectorization 

y=df.iloc[:,-1].values

################################################################ Applying Naive Bayes #######################################################
# X,y
# Training set
# Test Set(Already know the result)
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

clf1=GaussianNB()
clf2=MultinomialNB()
clf3=BernoulliNB()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)

y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)
y_pred3=clf3.predict(X_test)


from sklearn.metrics import accuracy_score

print("Gaussian",accuracy_score(y_test,y_pred1))
print("Multinomial",accuracy_score(y_test,y_pred2))
print("Bernaulli",accuracy_score(y_test,y_pred3))

# Gaussian 0.7185
# Multinomial 0.8295
# Bernaulli 0.835

# Best accuracy is given by Bernauli 