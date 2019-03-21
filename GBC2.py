from sklearn.model_selection import train_test_split as tts
from sklearn.datasets  import  make_hastie_10_2  as mh
from sklearn.datasets import load_breast_cancer as lbc
from sklearn.metrics import accuracy_score
from AdaBoost import cut_value_array
from ID3 import SplitDatasets
from CART import predict
from BDT import  error  ##将quasi及leafValue函数改为##，似乎没多大影响
from numpy import *  ##起因是不知为何损失函数y有个系数2而不是1
import time

def quasi_residual(y_true,y_pred):
      return 2*y_true/(1+exp(2*y_true*y_pred))
##      return y_true/(1+exp(y_true*y_pred))

def threN(y,max_depth):
      return y/power(2,max_depth)

def leafValue(y,K):
      if K==2:
            return sum(y)/sum(abs(y)*(2-abs(y)))
##            return sum(y)/sum(abs(y)*(1-abs(y)))
      else:
            return (K-1)/K*sum(y)/sum(abs(y)*(1-abs(y)))

##基于最小平方和划分数据
def bestSplit(X,y,threNum,K):
      if len(X)<2*threNum or len(unique(y))==1:  ##至少是阈值的两倍
            return None,leafValue(y,K)
      m,n=shape(X)
      bestErr=inf; feature=inf; value=inf
      for i in range(n):
            values=cut_value_array(X[:,i])
            for split_value in values:
                  index1,index2=SplitDatasets(i,split_value,X)
                  ##问题在于划分到某个阶段以后X无论怎么分都无法达到下面的要求
                  ##也就是下面这块根本一次都没执行，直接返回inf,inf...
                  if len(index1)<threNum or len(index2)<threNum:  ##数据过少
                        continue
                  tempErr=error(y[index1])+error(y[index2])
                  if tempErr<bestErr:
                        feature=i
                        value=split_value
                        bestErr=tempErr
      if bestErr==inf:  ##压根没变
            return None,leafValue(y,K)
      return feature, value

def createTree(X,residual,max_depth,threNum,K):
      feature, value=bestSplit(X,residual,threNum,K)
      if feature==None:
            return value
      Tree={'featIndex':feature,'value':value}
      left,right=SplitDatasets(feature,value,X)
      if max_depth>1:
            Tree['left']=createTree(X[left],residual[left],max_depth-1,threNum,K)
            Tree['right']=createTree(X[right],residual[right],max_depth-1,threNum,K)
      else:
            Tree['left']=leafValue(residual[left],K)
            Tree['right']=leafValue(residual[right],K)
      return Tree

def influTrim(y,alpha=0.2):
      w=abs(y)*(2-abs(y))
      wT=sorted(w)
      totalSum=sum(wT)
      thisSum=0
      for i in range(len(wT)):
            thisSum+=wT[i]
            if thisSum>alpha*totalSum:
                  break
      print(wT[i])
      return where(w>=wT[i])[0]

def train(X,y,max_depth,n_estimators,learning_rate,subsample=1.0,K=2,siGn=0):
      Trees=[]
      m=shape(X)[0]
      F0=0.5*log((1+mean(y))/(1-mean(y)))
      Fx=ones(m)*F0
      threNum=threN(m,max_depth)  ##叶节点包含的最小数据个数
      while n_estimators>0:
            residual=quasi_residual(y,Fx)  ##更新残差
            if subsample<1.0:  ##执行亚采样
                  index=random.choice(range(m),size=int(subsample*m),replace=False)
            else:
                  index=array(range(m))
            if siGn==1:   ##影响力剪枝
                  index=influTrim(residual[index])
            tree=createTree(X[index],residual[index],max_depth,threNum,K)
            thisPred=[]  ##本轮预测
            Trees.append(tree)
            for i in range(m):
                  thisPred.append(predict(tree,X[i],learning_rate))
            Fx+=array(thisPred)
            n_estimators-=1
      return Trees

def test(X,Trees,learning_rate):
      m=shape(X)[0]
      n=len(Trees)
      pred=zeros(m)
      for i in range(m):
            for j in range(n):
                  pred[i]+=predict(Trees[j],X[i],learning_rate)
      pred[pred>0]=1
      pred[pred<=0]=-1
      return pred

##剔除低"影响力"的数据，速度能提升10%，精度略有下降。再加上亚采样，节省更多。
if __name__=='__main__':
      
      start=time.time()
      
##      Data=lbc();       X=Data.data;      y=Data.target;          y[y==0]=-1
      data=mh();        X=data[0];        y=data[1]
      X_train, X_test, y_train, y_test=tts(X,y,random_state=1120,test_size=0.2,stratify=y)
      Trees=train(X_train,y_train,7,80,0.1,subsample=1,siGn=0)
      pred=test(X_test,Trees,0.1)
      print(accuracy_score(pred,y_test))
      end=time.time()
      print('Running time:',end-start)























