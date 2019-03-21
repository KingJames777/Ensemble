from numpy import *

def R2_score(pred,y):
      u=((y - pred) ** 2).sum() 
      v=((y- y.mean(axis=0)) ** 2).sum()
      return 1-u/v

##注意这里X和y放在一起
def load_data(filename):
      n=len(open(filename).readline().split('\t'))
      data=[]
      for line in open(filename).readlines():
            lineArr=[]
            curLine=line.strip().split('\t')
            for i in range(n):
                  lineArr.append(float(curLine[i]))
            data.append(lineArr)
      return array(data)

def predict(Tree,test,learning_rate):
      tree=Tree.copy()
      while isTree(tree):  ##要学会.
            if test[tree['featIndex']]<=tree['value']:
                  tree=tree['left']
            else:
                  tree=tree['right']
      return tree*learning_rate

def isTree(obj):
      return type(obj).__name__=='dict'
      












