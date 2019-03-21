from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from CART import load_data
from sklearn import preprocessing
import pandas as pd
from numpy import *

def bestParas(X_train,y_train):
      pipeline=make_pipeline(preprocessing.StandardScaler(),GBR())
      hyperparameters={'gradientboostingregressor__learning_rate':[0.1,0.2],
                       'gradientboostingregressor__max_depth':[3,6,9],
                       'gradientboostingregressor__n_estimators':[50,80],
                       'gradientboostingregressor__subsample':[0.8,0.9,1.0]}
      gbr=GridSearchCV(pipeline,hyperparameters,cv=10).fit(X_train,y_train)
      return gbr

def trainTestSplit(data):
      m,n=shape(data)
      index_test=random.choice(range(m),size=int(m/5),replace=False)
      index_train=list(set(range(m))-set(index_test))
      return index_train,index_test

def trainMSEtest(data,index_train,index_test):
      X=data[:,:-1];   y=data[:,-1]
      X_train=X[index_train]    ;     X_test=X[index_test]
      y_train=y[index_train]    ;      y_test=y[index_test]
      gbr=GBR(loss='ls',max_depth=6,n_estimators=80,subsample=0.6,learning_rate=0.08).fit(X_train,y_train)
      pred_test=gbr.predict(X_test)
      pred_train=gbr.predict(X_train)
      print(around(pred_test[:30]),'\n',y_test[:30])
      print('Test MSE:',mse(y_test, pred_test))
      print('Train MSE:',mse(y_train, pred_train))


if __name__ == '__main__':
      random.seed(20)
      data=load_data('abalone.txt')
      index_train,index_test=trainTestSplit(data)
      trainMSEtest(data,index_train,index_test)





















