from sklearn.datasets import load_breast_cancer as lbc
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from numpy import *
from ROC import roc

def cut_value_array(X,points=10):
      low,up=X.min(),X.max()
      res=zeros(points)
      step=(up-low)/points
      res[0]=low+step/2
      for i in range(1,points):
            res[i]=res[i-1]+step
      return res

def classifier(X,cut_point,flag):
      pred=ones(len(X))
      if flag==0:
            pred[X<=cut_point]=-1
      else:
            pred[X>cut_point]=-1
      return pred

##输出树桩的特征编号以及对应的划分值，误差率
def DecisionStump(X,y,weight,points):
      m,n=X.shape;      stump={};         error=inf
      for i in range(n):
            cut_points=cut_value_array(X[:,i],points)
            for j in range(points):
                  for flag in [0,1]:
                        pred=classifier(X[:,i],cut_points[j],flag)
                        temp_error=sum(weight*(pred!=y))  ##  准确率当成错误率！
                        if temp_error<error:
                              error=temp_error
                              stump['value'],stump['featNum']=cut_points[j],i 
                              stump['inequa'],stump['pred']=flag,pred  ##忘记更新！
      stump['alpha']=0.5*log(1/error-1)
      return stump

def train(X,y,n_estimators,points=10):
      m=shape(X)[0];    weight=ones(m)/m;       stumps=[];        error=inf
      while n_estimators>0:
            stump=DecisionStump(X,y,weight,points)
            pred=stump.pop('pred')
            stumps.append(stump)
            
            temp=exp(-stump['alpha']*y*pred)  ##更新样本权重
            weight*=temp
            temp=sum(weight)
            weight/=temp
            n_estimators-=1
      return stumps

def predict(X,stumps):
      m,n=len(X),len(stumps)
      pred=zeros(m)
      for i in range(n):
            pred+=stumps[i]['alpha']*classifier(X[:,stumps[i]['featNum']],
                                                stumps[i]['value'],stumps[i]['inequa'])
      return pred,sign(pred)

if __name__=='__main__':
      Data=lbc();       X=Data.data;            y=Data.target;          y[y==0]=-1
      X_train, X_test, y_train, y_test=tts(X,y,random_state=1990,test_size=0.2,stratify=y)

      stumps=train(X_train,y_train,50,15)  ##特征划分也是要考虑的超参数
      prob,pred=predict(X_test,stumps)
      print(accuracy_score(pred,y_test))
      roc(prob,y_test)
      




















