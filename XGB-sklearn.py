from sklearn.datasets  import  load_digits as ld
from sklearn.model_selection import train_test_split as tts
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance
from numpy import *

def classify():
    data = ld()
    X=data.data;    y=data.target
    X_train, X_test, y_train, y_test=tts(X,y,random_state=1990,test_size=0.2,stratify=y)
##gamma: threhold of gain,deciding whether to split any further.
#min_child_weight: minimum weights of a leaf, similar to the minimum num of instances in a leaf
##reg_lambda: L2 norm for the weight of each instance
##base_score: positive prob threhold
    clf = xgb.XGBClassifier(objective='multi:softmax',reg_lambda=1,min_child_weight=1,
                        gamma=1,n_estimators=200,max_depth=6,seed=2010,learning_rate=0.02)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    ##fig,ax = plt.subplots(figsize=(10,15))
    ##plot_importance(clf,height=0.5,max_num_features=64,ax=ax)
    ##plt.show()

def load_data(filename):
      n=len(open(filename).readline().split('\t'))-1  ##属性数，即列数
      X=[];y=[]
##      skip=['1','2','24','25','26','27','29']
      for line in open(filename).readlines():
            lineArr=[]
            curLine=line.strip().split('\t')
##            if curLine[-1] in skip:
##                  continue
            for i in range(n):
                  lineArr.append(float(curLine[i]))
            X.append(lineArr)
            y.append(int(curLine[-1]))
      return array(X),array(y)

def regression():
    filename='abalone.txt'
    X,y=load_data(filename)
    clf=xgb.XGBRegressor(reg_lambda=3,min_child_weight=2,
                        gamma=1,n_estimators=30,max_depth=5,seed=2010)
    clf.fit(X,y)
    print(around(clf.predict(X[:18])),'\n',y[:18])

if __name__=='__main__':
    classify()








