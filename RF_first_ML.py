import numpy as np
import pandas as pd #数据导入导出工具
from sklearn.model_selection import train_test_split
from sklearn import preprocessing #预处理函数
from sklearn.ensemble import RandomForestRegressor #随机森林
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV #这两个是交叉验证用的
from sklearn.metrics import mean_squared_error, r2_score #评估函数：均方差和回归误差
from sklearn.externals import joblib #这个是保存数据用的

#加载数据
dataset_url='http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url,sep=';') #返回DataFrame类型；加上sep参数后数据好看多了

#分别显示前五组数据，数据规模以及统计特征
print(data.head(),data.shape,data.describe()) 

#分离标记和输入数据
y=data.quality #标记，quality是特征之一
X=data.drop('quality',axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1990,stratify=y)

##预处理，设置交叉验证pipeline,这个pipeline已经是个模型，可以用来拟合数据了。
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

##以字典形式设定所需超参数，关键字格式要和打印出的一样，即加上'random...'
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#保存模型以备再次调用
joblib.dump(clf, 'rf_regressor.pkl')
