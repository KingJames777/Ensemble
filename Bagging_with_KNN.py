from sklearn.datasets import load_breast_cancer as lbc
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from numpy import *

##结论，kNN的Bagging没啥效果。

##有放回采样n个数据集
def bootstrap(m,n):
      samples=zeros((n,m),dtype='int8')
      for i in range(n):
            samples[i]=random.choice(range(m),size=m)  ##每个数据集对应的索引
      return samples

##测试点X与所有训练数据的欧几里得距离
def dist(X_train,X):
      a=X_train-X
      b=a*a
      return b.sum(axis=1)

def kNN(X_train,X_test,y_train,k=3):
      pred=[]
      for i in range(len(X_test)):
            distance=dist(X_train,X_test[i])
            res=[]
            for j in range(k):
                  index=distance.argmin()
                  res.append(y_train[index])
                  delete(distance,index)
            classes,counts=unique(res,return_counts=True)
            pred.append(classes[counts.argmax()])
      return pred

def voting(X_train, X_test, y_train):
      n=11
      samples=bootstrap(len(X_train),n)
      pred=[]
      for i in range(n):
            pred.append(kNN(X_train[samples[i]],X_test,y_train[samples[i]]))
      column_sum=sum(pred,axis=0)
      return sign(column_sum)

if __name__=='__main__':
      Data=lbc()
      X=Data.data
      X=(X-X.mean(axis=0))/X.std(axis=0)
      y=Data.target
      y[y==0]=-1
      X_train, X_test, y_train, y_test=tts(X,y,random_state=1,test_size=0.2,stratify=y)

      pred=kNN(X_train,X_test,y_train)
      print('Bagging前:',accuracy_score(pred,y_test))

      pred=voting(X_train, X_test, y_train)
      print('Bagging后:',accuracy_score(pred,y_test))






















