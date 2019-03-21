from sklearn.model_selection import train_test_split as tts
from sklearn.datasets  import  load_digits as ld
import lightgbm as lgb
from numpy import *

data = ld();        X=data.data;        y=data.target
X_train, X_test, y_train, y_test=tts(X,y,random_state=2020,test_size=0.2,stratify=y)

clf=lgb.LGBMClassifier(min_data_in_leaf=10,num_leaves=30,colsample_bytree=0.4,
                       max_depth=7,subsample=0.8,
                       learning_rate=0.01,n_estimators=500,).fit(X_train,y_train)
print(clf.score(X_test,y_test))




