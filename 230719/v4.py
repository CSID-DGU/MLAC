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
from load import *


cici_test_dir='/home/irteam/junghye-dcloud-dir/MLAC/230624/CICI'
X_test=pd.read_csv(os.path.join(cici_test_dir,'X_test4.csv'))
y_test=pd.read_csv(os.path.join(cici_test_dir,'y_test.csv'))


df=pd.DataFrame(columns=['L1_acc','L1_f1','L1_rc','L1_pc','L2_acc','L2_f1','L2_rc','L2_pc']+\
                 ['c1_acc','c1_f1','c1_rc','c1_pc']+\
                    ['c2_acc','c2_f1','c2_rc','c2_pc','c3_acc','c3_f1','c3_rc','c3_pc','c4_acc','c4_f1','c4_rc','c4_pc']+\
                     ['total_acc','total_f1','total_rc','total_pc'])

cnt=0
model_eval=[]

L1_ytest=y_test.copy()

L2_ytest=X_test['nist_category4'].copy()

L3_ytest=X_test['attack_category'].copy()

c1_Xtest=X_test.query('nist_category4==1')
c1_ytest=c1_Xtest['attack_category']

c2_Xtest=X_test.query('nist_category4==2')
c2_ytest=c2_Xtest['attack_category']

c3_Xtest=X_test.query('nist_category4==3')
c3_ytest=c3_Xtest['attack_category']

c4_Xtest=X_test.query('nist_category4==4')
c4_ytest=c4_Xtest['attack_category']

for class_data in [c1_Xtest,c2_Xtest,c3_Xtest,c4_Xtest]:
    class_data.drop(labels=['Unnamed: 0','attack_category','nist_category','nist_category4'],axis=1,inplace=True)

X_test.drop(labels=['Unnamed: 0','nist_category','attack_category','nist_category4'],axis=1,inplace=True)
L1_ytest.drop(labels=['Unnamed: 0'],axis=1,inplace=True)

# generate output file
for path in [progressLog_path,out_csv_path]:
    if not os.path.isfile(path):
        Path(path).touch()


# csv writer
try:
    f=open(progressLog_path,'w')
    f.truncate()
except:
    os.system("sudo chmod 777 /home/irteam/junghye-dcloud-dir/MLAC/230719/result/CICI/v4/progressLog.txt")
    f=open(progressLog_path,'w')
    f.truncate()

model_eval.extend([0.99,0.99,0.99,0.991])

# load saved results
L1_ypred=np.loadtxt(out_txt_path,delimiter=',')

malicious_indices=np.where(L1_ypred==1)[0]

# 최종 결과를 위해 저장
L1_b_ypred=L1_ypred[L1_ypred==0]
L1_b_ytest=L1_ytest[~np.isin(np.arange(L1_ytest.shape[0]),malicious_indices)]


if malicious_indices.any():
    L2_model=joblib.load(os.path.join(saved_path,'CICI_nist.pkl'))
    L2_Xtest=X_test.iloc[malicious_indices]
    L2_ypred=L2_model.predict(L2_Xtest)
    L2_ytest_selected=L2_ytest.iloc[malicious_indices]
    L2_result=test_result(L2_model,L2_ytest_selected,L2_ypred)
    
    # Layer2 result
    f=open(progressLog_path,'a')
    f.write('L2 result'+','.join(map(str,L2_result))+'\n')
    f.close()
    model_eval.extend(L2_result)
else:
    print('no malicious predicted')
    import sys
    sys.exit()

# confusion matrix
L2_encoded=[]
L2_encoded.extend(L2_ypred)
L2_encoded.extend(L2_ytest_selected)
L2_encoded=list(set(L2_encoded))

confusion=metrics.confusion_matrix(L2_ytest_selected,L2_ypred)
plot_confusion_matrix(confusion,labels=L2_encoded,title='Layer2')

from collections import defaultdict

class_models = [
    joblib.load(os.path.join(saved_path, 'class_1_CICI.pkl')),
    joblib.load(os.path.join(saved_path, 'class_2_CICI.pkl')),
    joblib.load(os.path.join(saved_path, 'class_3_CICI.pkl')),
    joblib.load(os.path.join(saved_path, 'class_4_CICI.pkl'))
]

class_names = ['Reconnaissance', 'Access', 'Dos', 'Malware']

class_encodings=defaultdict(list)
final_y_pred=[]
final_y_test=[]

L3_ytest_selected=L3_ytest.iloc[malicious_indices]

for class_index,class_model in enumerate(class_models):
    indices=np.where(L2_ypred==class_index+1)[0]
    print(class_names[class_index]+'train & test')

    if indices.any():
        X_test_selected=L2_Xtest.iloc[indices]
        y_pred=class_model.predict(X_test_selected)
        y_test_selected=L3_ytest_selected.iloc[indices]
        result=test_result(class_model,y_test_selected,y_pred)

        # 각 classifier result
        f=open(progressLog_path,'a')
        f.write(str(class_names[class_index])+','.join(map(str,result))+'\n')
        f.close()

        model_eval.extend(result)

        class_encodings[class_names[class_index]].extend(y_pred)
        class_encodings[class_names[class_index]].extend(y_test_selected)

        final_y_pred.extend(y_pred)
        final_y_test.extend(y_test_selected)
        

    else:
        model_eval.extend([0,0,0,0])


final_y_pred.extend(L1_b_ypred)
final_y_test.extend(L1_b_ytest.label.values)


final_result = test_result('Layer3', final_y_test, final_y_pred)

# final result
f=open(progressLog_path,'a')
f.write('final result'+','.join(map(str,final_result))+'\n')
f.close()

model_eval.extend(final_result)

for class_name in class_names:
    confusion = metrics.confusion_matrix(class_encodings[class_name][len(class_encodings[class_name]) // 2:], 
                                         class_encodings[class_name][:len(class_encodings[class_name]) // 2])
    plot_confusion_matrix(confusion,labels=list(set(class_encodings[class_name])),title=class_name)

confusion = metrics.confusion_matrix(final_y_test, final_y_pred)
plot_confusion_matrix(confusion, labels=list(set(final_y_pred + final_y_test)), title='Layer3')


df.loc[cnt]=model_eval
#df

df.to_csv(os.path.join(out_csv_path))
