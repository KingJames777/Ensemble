from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from ID3 import CreateDecisionTree , Classifier
from Bagging_with_KNN import bootstrap
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from numpy import *

def attribute_selection(m,n,k):  ##从n个属性中选择k个，执行m次
      samples=zeros((m,k),dtype='int8')
      for i in range(m):
            samples[i]=random.choice(range(n),size=k,replace=False)  ##每个数据集对应的索引
      return samples

def voting(X):
      n=shape(X)[1]
      pred=[]
      for i in range(n):
            values,counts=unique(X[:,i],return_counts=True)
            counts=list(counts)
            pred.append(values[counts.index(max(counts))])
      return pred

if __name__=='__main__':
      wine=load_wine()
      X=wine.data
      y=wine.target
      X_train, X_test, y_train, y_test=tts(X,y,random_state=1990,test_size=0.2,stratify=y)
      
      m,n=X_train.shape
      precision=[]
      for n_classifier in range(1,20):
            
            k=int(log2(n))  ##随机抽取的属性数
            data_idx=bootstrap(m,n_classifier)  ##  n_classifier*m
            attr_idx=attribute_selection(n_classifier,n,k)   ## n_classifier*k
            
            pred=[]
            for i in range(n_classifier):
                  epsilon=1e-3
                  DT=CreateDecisionTree((X_train[data_idx[i]])[:,attr_idx[i]],y_train[data_idx[i]],epsilon)
                  pred.append(Classifier(DT,X_test[:,attr_idx[i]]))
            pred=array(pred)

            pred=voting(pred)
            precision.append(accuracy_score(pred,y_test))
      
      plt.plot(range(1,20),precision)
      plt.show()















