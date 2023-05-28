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
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

"""
Devide Data into Train & Test set
Input : type(binary VS multiclass)
Ouput : X_train, X_test, y_train, y_test set
Bring Preprocessed Data
"""
def TrainTestSplit(type, data):
    print('Dataset Split')
    if type == 'B':
        print('Binary Classification Dataset Split')
        target = data['label']
    else:
        # Remove Benign Data
        benign = data[data['label'] == 0].index
        data.drop(benign, inplace=True)
        target = data['attack_category']

    #data.drop(labels=['attack_category, label'], axis=1, inplace=True)
    del data['attack_category']
    del data['label']
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
Plot Confusion Matrix
"""
# confusion matrix plot
def plot_confusion_matrix(con_mat,labels,title:str,cmap=plt.cm.get_cmap('Blues'),normalize=False, confusion_path=None):
    plt.imshow(con_mat,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks=np.arange(len(labels))
    nlabels=[]
    for k in range(len(con_mat)):
        n=sum(con_mat[k])
        nlabel='{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)

    plt.xticks(marks,labels,rotation=45)
    plt.yticks(marks,nlabels)

    thresh=con_mat.max()/2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    #이미지 저장
    plt.savefig(confusion_path+'/'+title+'.png',facecolor='#eeeeee',edgecolor='blue',pad_inches=0.5)
    plt.clf()


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
        os.makedirs(os.getcwd()+'/Binary/Matrix',exist_ok=True)
        # marks=np.arange(len(len(y_train['label'].value_counts())))
        plot_confusion_matrix(confusion,labels=marks,title=name,confusion_path=os.getcwd()+'/Binary/Matrix')           
    accuracy = accuracy.round(3)
    accuracy.to_csv(os.getcwd()+file+'.csv',index=False)


"""
Multiclass Classification
Input : file name, model list(from getModels function), train&test set(from getData function)
Output : Evaluation result as csv file, Confusion Matrix of each Model test
"""
def MultiClassification(file, models, X_train, X_test, y_train, y_test):
    accuracy = pd.DataFrame(columns=['Model','Acc','F1_mi','Recall_mi','Precision_mi','F1_ma','Recall_ma','Precision_ma','F1_we','Recall_we','Precision_we','Execution'])
    print('Model\tAcc\tF1_mi\tRecall_mi\tPrecision_mi\tF1_ma\tRecall_ma\tPrecision_ma\tF1_we\tRecall_we\tPrecision_we\tExecution')
    cnt = 0
    for name, model in models:
        start_time = time.time()
        # 모델 훈련 및 예측
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        # 지표 추출
        delta = time.time() - start_time
        acc = accuracy_score(y_test, prediction)
        f1_mi = f1_score(y_test, prediction,average='micro')
        f1_ma = f1_score(y_test, prediction,average='macro')
        f1_we = f1_score(y_test, prediction,average='weighted')
        recall_mi = recall_score(y_test, prediction, average='micro')
        recall_ma = recall_score(y_test, prediction, average='macro')
        recall_we = recall_score(y_test, prediction, average='weighted')
        precision_mi = precision_score(y_test, prediction, average='micro')
        precision_ma = precision_score(y_test, prediction, average='macro')
        precision_we = precision_score(y_test, prediction, average='weighted')
        confusion = metrics.confusion_matrix(y_test, prediction)
        # 저장
        accuracy.loc[cnt] = [name, acc, f1_mi, f1_ma, f1_we, recall_mi, recall_ma, recall_we, precision_mi, precision_ma, precision_we, delta]
        print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f} secs'.format(name, acc, f1_mi, recall_mi, precision_mi, f1_ma, recall_ma, precision_ma, f1_we, recall_we, precision_we, delta))
        cnt += 1
        matrix = open('/home/irteam/wendyunji-dcloud-dir/wendyunji/2023-1/MLAC/Classification/result/230330/matrix/'+file+'_'+name+'.txt','w')
        matrix.write(str(confusion))            
    accuracy = accuracy.round(3)
    accuracy.to_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/2023-1/MLAC/Classification/result/230330/'+file+'.csv',index=False)

    