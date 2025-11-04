# we are using statsmodel library which is used regression analysis
import statsmodels.api as sm
import pandas as pd  

url="https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
data=pd.read_csv(url,index_col=0)

# Define the independent variables (add a constant for the intercept)
X = data[['TV', 'Radio', 'Newspaper']]
X = sm.add_constant(X)

# Define the dependent variable
y = data['Sales']

# Fit the model using the independent and dependent variables
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary()) 