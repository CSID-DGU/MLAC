import time
    
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics

import os
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from load_24class import *

cici_test_dir='/home/irteam/dcloud-global-dir/MLAC/2308/CICI'
X_train=pd.read_csv(os.path.join(cici_test_dir,'X_train.csv'))
y_train=pd.read_csv(os.path.join(cici_test_dir,'L3_train.csv'))

X_test=pd.read_csv(os.path.join(cici_test_dir,'X_test.csv'))
y_test=pd.read_csv(os.path.join(cici_test_dir,'L3_test.csv'))


df=pd.DataFrame(columns=['model','acc','f1','rc','pc'])

cnt=0
model_eval=[]


# Define Models
models = []
models.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=5, max_features=3)))    
models.append(('CART', DecisionTreeClassifier(max_depth=5)))
models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200)))
models.append(('ABoost', AdaBoostClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('MLP', MLPClassifier()))

# generate output file

for path in [progressLog_path,out_csv_path]:
    if not os.path.isfile(path):
        Path(path).touch()

# 
# csv writer
try:
    f=open(progressLog_path,'w')
    f.truncate()
except:
    os.system("sudo chmod 777 {progressLog_path}")
    f=open(progressLog_path,'w')
    f.truncate()

for name,model in models:
    
    model.fit(X_train,y_train)
    print(f'{name} test starts...')

    y_pred=model.predict(X_test)

    model_eval=[]
    model_eval.append(name)

    eval_result=test_result(name,y_test,y_pred)
    model_eval.extend(eval_result)

    f=open(progressLog_path,'a')
    f.write(str(model_eval[0])+','.join(map(str,model_eval[1:]))+'\n')
    confusion=metrics.confusion_matrix(y_test,y_pred)

    plot_confusion_matrix(confusion,labels=list(set(y_test.attack_category)),title=name)
    df.loc[cnt]=model_eval

    cnt+=1

df.to_csv(os.path.join(out_csv_path))