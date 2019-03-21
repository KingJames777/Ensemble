##'prob:array-like'

import matplotlib.pyplot as plt
from numpy import *

## TPR=TP/(TP+FN)     FPR=FP/(FP+TN)
def roc(prob,y):
      TPR,FPR=[],[]
      t1=-sort(-prob)  ##降序这么操作!!!
      thre=append(inf,t1)
      m=len(y)
      area=0
      for j in range(len(thre)):
            TP,FP,TN,FN=0,0,0,0
            pred=sign(prob-thre[j]+1e-12)
            for i in range(m):
                  if pred[i]==1 and y[i]==1:
                        TP+=1
                  elif pred[i]==1 and y[i]==-1:
                        FP+=1
                  elif pred[i]==-1 and y[i]==-1:
                        TN+=1
                  else:
                        FN+=1
            TPR.append(TP/(TP+FN))
            FPR.append(FP/(FP+TN))
            if j>0:
                  area+=(FPR[j]-FPR[j-1])*TPR[j]
      print('auc:',area)
      plt.plot(FPR,TPR)
      plt.show()
