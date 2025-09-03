
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import random 

X,y = load_diabetes(return_X_y=True)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Making our own class of gd 
class GDRegressor:
    def __init__(self,learning_rate,epochs,batch_size):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size=batch_size
        
    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1]) #initialising all coefecients with 1 
        
        for i in range(self.epochs): # no of iterations 
            for j in range(X_train[0]/self.batch_size) : # this loop will run uptill total_number of batches 
                
                # randomly select row 
                indx = random.sample(range(X_train.shape[0]),self.batch_size) # random no bw range of rows ... and generate self.batch_size .no's of element

                # update all the coef and the intercept
                y_hat = np.dot(X_train[indx],self.coef_) + self.intercept_
                
                
                intercept_der = -2 * np.mean(y_train[indx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                coef_der = -2 * np.dot((y_train[indx] - y_hat),X_train[indx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
        
        print(self.intercept_,self.coef_)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_
    
    
mb_gd = GDRegressor(epochs=50, learning_rate=0.01, batch_size=int(X_train.shape[0] / 10))

