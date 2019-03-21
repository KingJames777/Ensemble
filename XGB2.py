from sklearn.model_selection import train_test_split as tts
from sklearn.datasets  import  make_hastie_10_2  as mh
from sklearn.datasets import load_breast_cancer as lbc
from sklearn.metrics import accuracy_score
from AdaBoost import cut_value_array
from ID3 import SplitDatasets
from CART import predict
from GBC2 import test
from numpy import *
import time

def GraHes(y,F):
      temp=exp(y*F)
      g=-y/(1+temp)
      return g,g*g*temp

def twoGH(grad,hes,index1,index2,lamda):
      G1=sum(grad[index1]);         H1=sum(hes[index1])
      G2=sum(grad[index2]);         H2=sum(hes[index2])
      return G1*G1/(H1+lamda)+G2*G2/(H2+lamda)

def bestSplit(X,grad,hes,lamda,eta):
      m,n=shape(X)
      if m<2:  ##防止所剩元素低于2的情况
            return None,inf
      bestErr=-inf; feature=inf; value=inf
      for i in range(n):
            values=cut_value_array(X[:,i])
            for split_value in values:
                  index1,index2=SplitDatasets(i,split_value,X)
                  temp=twoGH(grad,hes,index1,index2,lamda)
                  if temp>bestErr:
                        bestErr=temp
                        feature=i
                        value=split_value
      return feature, value

def createTree(X,max_depth,grad,hes,lamda,eta):
      feature, value=bestSplit(X,grad,hes,lamda,eta)
      G=sum(grad);      H=sum(hes)
      if feature==None:
            return -G/(H+lamda)
      left,right=SplitDatasets(feature,value,X)
      if (twoGH(grad,hes,left,right,lamda)-G*G/(H+lamda))/2-eta<=0:
            return -G/(H+lamda)
      Tree={'featIndex':feature,'value':value}
      if max_depth>1:
            Tree['left']=createTree(X[left],max_depth-1,grad[left],hes[left],lamda,eta)
            Tree['right']=createTree(X[right],max_depth-1,grad[right],hes[right],lamda,eta)
      else:
            G1=sum(grad[left]);         H1=sum(hes[left])
            G2=sum(grad[right]);          H2=sum(hes[right])
            Tree['left']=-G1/(H1+lamda);    Tree['right']=-G2/(H2+lamda)
      return Tree

def train(X,y,max_depth,n_estimators,learning_rate,lamda,eta):
      Trees=[]
      m=shape(X)[0]
      F=zeros(m)
      while n_estimators>0:
            grad,hes=GraHes(y,F)
            tree=createTree(X,max_depth,grad,hes,lamda,eta)
            pred=[]
            Trees.append(tree)
            for i in range(m):
                  pred.append(predict(tree,X[i],learning_rate))
            F+=array(pred)
##            print(F[:10])
            n_estimators-=1
      return Trees


if __name__=='__main__':
      
      start=time.time()
      
##      Data=lbc();       X=Data.data;      y=Data.target;          y[y==0]=-1
      data=mh();        X=data[0];        y=data[1]
      X_train, X_test, y_train, y_test=tts(X,y,random_state=1990,test_size=0.2,stratify=y)
      Trees=train(X_train,y_train,5,40,0.1,0,0.8)
      pred=test(X_test,Trees,0.1)
      print(accuracy_score(pred,y_test))
      
      end=time.time()
      print('Running time:',end-start)













