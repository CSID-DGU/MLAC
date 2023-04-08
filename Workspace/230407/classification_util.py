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
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
import os


"""
Devide Data into Train & Test set
Input : type(binary VS multiclass)
Ouput : X_train, X_test, y_train, y_test set
Bring Preprocessed Data
"""
def TrainTestSplit(data):
    target = data['dlabel']
    data.drop(labels=['dlabel'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=True, stratify=target, random_state=34)
    return X_train, X_test, y_train, y_test

"""
Loading various Machine Learning model from sckit-learn library
Input : X
Output : Various ML models in list
"""
def getModels():
    models = []
    models.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=5, max_features=3)))    
    models.append(('DT', DecisionTreeClassifier(max_depth=5)))
    models.append(('NB', GaussianNB()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('QDA', QuadraticDiscriminantAnalysis()))
    models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200)))
    models.append(('ABoost', AdaBoostClassifier()))
    models.append(('k-NN', KNeighborsClassifier()))
    models.append(('MLP', MLPClassifier()))
    models.append(('SVM', svm.LinearSVC()))
    return models

"""
Binaryclass Classification
Input : file name, model list(from getModels function), train&test set(from getData function)
Output : Evaluation result as csv file, Confusion Matrix of each Model test
Evaluation Result : accuracy, f1 score, recall, precision, excution time of each model
"""
def BinaryClassification(file, models, X_train, X_test, y_train, y_test):
    accuracy = pd.DataFrame(columns=['Model','Acc','F1','Recall','Precision','Execution'])
    print('Model\tAcc\tF1\tRecall\tPrecision\tExecution')
    cnt = 0
    for name, model in models:
        start_time = time.time()
        # 모델 훈련 및 예측
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        # 지표 추출
        delta = time.time() - start_time
        acc = accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        confusion = metrics.confusion_matrix(y_test, prediction)
        # 저장
        accuracy.loc[cnt] = [name, acc, f1, recall, precision, delta]
        print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f} secs'.format(name, acc, f1, recall, precision, delta))
        cnt += 1        
    accuracy = accuracy.round(3)
    accuracy.to_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/2023-1/MLAC/Workspace/230407/'+file+'.csv',index=False)


