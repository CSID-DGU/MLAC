import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
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
from load import create_pipeline
import os

# 전처리
# Get Dataset
files={
    'CICI':'/home/irteam/junghye-dcloud-dir/MLAC/data/encoded_ConcatedCICI.csv',
    'UNSW': '/home/irteam/junghye-dcloud-dir/MLAC/data/encoded_ConcatedUNSW.csv'
}

data = pd.read_csv(files['CICI'])
#target = data['attack_category']
binary_target=data['label']
multiclass_labels_1=data['nist_category']
multiclass_labels_2=data['attack_category']
class_1_data=data[data['nist_category']==0]
class_2_data=data[data['nist_cateogory']==1]
class_3_data=data[data['nist_category']==2]
class_4_data=data[data['nist_category']==3]
data=data.drop(labels=['label','attack_category','nist_category'],axis=1)






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
"""
binary -> positive인 것에 대해서만 multiclass 분류 진행
4 classifier, 15 classifier는 다른 분류기를 사용
대신 4 class로 분류된 각 subclass에 대해서 분류기 4개를 만들어서 돌리기..??? 그럼 성능이 어떻게 나오는지 

"""

df=pd.Dataframe(columns=['name','binary_acc','binary_recall','multi_1_acc','binary_recall']+\
                 ['class_1_acc','class_1_recall','class_2_acc','class_2_recall','class_3_acc','class_3_recall','class_4_acc','class_4_recall'])
outpath='C:/Users/SAMSUNG/Documents/MLAC/result'
cnt=0
for name, model in models:
    
    # binary classification
    binary_model=create_pipeline(model)
   
    X_train,X_test,y_train,y_test=train_test_split(data,binary_target,test_size=0.3, shuffle=True, stratify=binary_target, random_state=34)
    binary_model.fit(X_train,y_train)
    #evaluation
    binary_pred=binary_model.predict(X_test)
    #evaluation result
    model_eval=[]
    model_eval.append(name)
    
    acc = accuracy_score(y_test, binary_pred)
    recall_ma = recall_score(y_test, binary_pred, average='macro')
    model_eval.append(acc)
    model_eval.append(recall_ma)

    #cm_1=confusion_matrix(y_test,binary_pred)
    #cm_df_1=pd.DataFrame(cm_1,index=binary_target,columns=binary_target)
    #cm_df_1.to_csv(os.path.join(outpath,'binary'),index=True,header=True)

    # malicious index 추출
    malicious_indices=np.where(binary_pred==1)[0]

    # multiclass_labels_1 추출
    multiclass_labels_1=multiclass_labels_1[malicious_indices]
    X_test=X_test[malicious_indices]

    multiclass_model_1=create_pipeline(model)

    multiclass_model_1.fit(X_train[malicious_indices],multiclass_labels_1)
    #evaluation
    multiclass_pred_1=multiclass_model_1.predict(X_test)
    #evaluation result
    
    
    acc = accuracy_score(y_test[malicious_indices], multiclass_pred_1)
    recall_ma = recall_score(y_test[malicious_indices], multiclass_pred_1, average='macro')
    model_eval.append(acc)
    model_eval.append(recall_ma)
    #cm_2=confusion_matrix(y_test[malicious_indices],multiclass_pred_1)
    #cm_df_2=pd.DataFrame(cm_2,index=multiclass_labels_1,columns=multiclass_labels_1)
    #cm_df_2.to_csv(os.path.join(outpath,'4class'),index=True,header=True)


    # multiclass_labels_2 추출
    multiclass_labels_2=multiclass_labels_2[malicious_indices]
    
    # 4개 클래스로 데이터를 분류 
    # x_train
    class_1_data=class_1_data[np.where(np.isin(class_1_data['attack_category'],multiclass_labels_2)==True)]
    class_2_data=class_2_data[np.where(np.isin(class_2_data['attack_category'],multiclass_labels_2)==True)]
    class_3_data=class_3_data[np.where(np.isin(class_3_data['attack_category'],multiclass_labels_2)==True)]
    class_4_data=class_4_data[np.where(np.isin(class_4_data['attack_category'],multiclass_labels_2)==True)]
    

    # 각 classifier

    for class_data in [class_1_data,class_2_data,class_3_data,class_4_data]:
        X_train,X_test,y_train,y_test=train_test_split(
            class_data.drop(labels=['nist_category','label','attack_category'],axis=1),
            class_data['attack_category'],
            test_size=0.3,
            random_state=42
        )

        classifier=create_pipeline(model)

        classifier.fit(X_train,y_train)

        classifier_pred=classifier.predict(X_test)

        acc = accuracy_score(y_test, classifier_pred)
        recall_ma = recall_score(y_test, classifier_pred, average='macro')
        model_eval.append(acc)
        model_eval.append(recall_ma)

    df.loc[cnt]=model_eval

df.to_csv(os.path.join(outpath,'evaluation'),header=True)



    