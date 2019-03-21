from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine as lw
from GBC2 import threN, createTree, bestSplit
from sklearn.metrics import accuracy_score
from CART import predict
from numpy import *
import time

def train(X,y,max_depth,n_estimators,learning_rate,K):
      Forest=[]  ##每个元素都是K个字典组成的列表
      m=shape(X)[0]
      Fx=zeros((m,K))
      threNum=threN(m,max_depth)
      Forcast=zeros((m,K))
      while n_estimators>0:
            Trees=[]
            eFx=exp(Fx)
            Px=eFx/sum(eFx,axis=1)[:,None]  ##按行求和
            residual=y-Px
            n_estimators-=1
            for k in range(K):  ##构造K棵树，分别拟合K类残差
                  Tree=createTree(X,residual[:,k],max_depth,threNum,K)
                  Trees.append(Tree)
                  for i in range(m):
                        Forcast[i,k]=predict(Tree,X[i],learning_rate)
            Fx+=Forcast
            Forest.append(Trees)
      return Forest

def test(X,Forest,learning_rate,K):
      m=shape(X)[0]
      n=len(Forest)
      pred=zeros((m,K))
      for i in range(m):
            for k in range(K):
                  for j in range(n):
                        pred[i,k]+=predict(Forest[j][k],X[i],learning_rate)
      return argmax(pred,axis=1)

def selectParas(X,y):
      acc,maxd,nest=0,0,0
      max_depth=range(3,7);   n_estimators=range(20,50,5);   l=0.1
      for m in max_depth:
            for n in n_estimators:
                  ac=[]
                  for r in range(5):
                        X_train, X_test, y_train, y_test=tts(X,y,random_state=r,test_size=0.2,stratify=y)
                        Forest=train(X_train,y_train,m,n,l,K)
                        pred=test(X_test,Forest,l,K)
                        ac.append(accuracy_score(pred,argmax(y_test,axis=1)))
                  mA=mean(ac)
                  print(mA)
                  if mA>acc:
                        acc=mA
                        maxd,nest=m,n
      return maxd,nest,acc
                              
            
if __name__=='__main__':
      start=time.time()
      Data=lw()
      X=Data.data       ; y=Data.target          ; K=len(unique(y))
      y=OneHotEncoder(sparse=False,dtype='int8').fit_transform(y.reshape(-1,1))

      max_depth,n_estimators,acc=selectParas(X,y)
      print(max_depth,'\t',n_estimators,'\t',acc)
      end=time.time()
      print('Running time:',end-start)







