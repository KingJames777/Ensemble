from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_regression as mr
from CART import load_data,R2_score,predict
from GBDT_sklearn import trainTestSplit
from ID3 import SplitDatasets
from numpy import *

def error(y):
      if len(y)==0:
            return 0
      return sum((y-mean(y))**2)

def best_split(X,y):
      if len(unique(y))==1:
            return None, mean(y)
      m,n=shape(X)
      bestErr=inf; feature=inf; value=inf
      for i in range(n):
            for split_value in set(X[:,i]):
                  index1,index2=SplitDatasets(i,split_value,X)
                  ##有的index全是False所以报错，在error处修改即可
                  tempErr=error(y[index1])+error(y[index2])
                  if tempErr<bestErr:
                        feature=i
                        value=split_value
                        bestErr=tempErr
      return feature, value

def createTree(X,y,depth):  
      feature, value=best_split(X,y)
      if feature==None:
            return value
      Tree={}
      Tree['featIndex']=feature
      Tree['value']=value
      left,right=SplitDatasets(feature,value,X)
      if depth>1:
            Tree['left']=createTree(X[left],y[left],depth-1)
            Tree['right']=createTree(X[right],y[right],depth-1)
      else:
            Tree['left']=mean(y[left])
            Tree['right']=mean(y[right])
      return Tree

def bdt(X,y,eps=1000,depth=6,lr=0.1):
      Tree=[]
      m,n=shape(X)
      residual=y.copy()  ## 残差
      while sum(residual**2)>eps:
            pred=[]
            tree=createTree(X,residual,depth)
            Tree.append(tree)
            for i in range(m):
                  pred.append(predict(tree,X[i]))
            residual-=array(pred)*lr  ##学习率
            print(sum(residual**2))
      print('Train MSE:',sum(residual**2)/len(X))
      return Tree

def validate(Tree,X_test,y_test):
      n=len(Tree)
      m=shape(X_test)[0]
      pred=zeros(m)
      for i in range(m):
            for j in range(n):
                  pred[i]+=predict(Tree[j],X_test[i])
      return R2_score(pred,y_test)

def predct(Tree,X_test,y_test,lr):
      n=len(Tree)
      m=shape(X_test)[0]
      pred=zeros(m)
      for i in range(m):
            for j in range(n):
                  pred[i]+=predict(Tree[j],X_test[i])*lr
      return pred

def CV(X_train,y_train,X_test,y_test):
      depth=range(5,10)
      eps=[1000,1500,2000,2500,3000]
      pr=-inf;  pd=inf;  pe=inf
      for d in depth:
            for ep in eps:
                  Tree=bdt(X_train,y_train,ep,d)
                  R=validate(Tree,X_test,y_test)
                  print(R)
                  if R>pr:
                        pr=R
                        pd=d
                        pe=ep
      return pd,pe

def trainMSEtest():
      data=load_data('abalone.txt')
      index_train,index_test=trainTestSplit(data)
      X=data[:,:-1];   y=data[:,-1]
      XX=X.copy()
      XX-=mean(X,axis=0)
      values,vectors=linalg.eig(XX.T.dot(XX))
      X=X.dot(vectors[:,:5])
      X_train=X[index_train]    ;     X_test=X[index_test]
      y_train=y[index_train]    ;      y_test=y[index_test]
      
      Tree=bdt(X_train,y_train,2500,6,0.1)
      pred=predct(Tree,X_test,y_test,0.1)
      print('Test MSE:',sum((pred-y_test)**2)/len(y_test))
      print(around(pred[:30]),'\n',y_test[:30])

if __name__ == '__main__': 
      X,y=mr(n_samples=1000,n_features=10,random_state=1,n_informative=6,noise=5,bias=3)
      index_train,index_test=trainTestSplit(X)
##      XX=X.copy()
##      XX-=mean(X,axis=0)
##      values,vectors=linalg.eig(XX.T.dot(XX))
##      X=X.dot(vectors[:,:6])
      X_train=X[index_train]    ;     X_test=X[index_test]
      y_train=y[index_train]    ;      y_test=y[index_test]
      Tree=bdt(X_train,y_train,10000,5,0.1)
      pred=predct(Tree,X_test,y_test,0.1)
      print('Test MSE:',sum((pred-y_test)**2)/len(y_test))
      print(around(pred[:30]),'\n',y_test[:30])


















