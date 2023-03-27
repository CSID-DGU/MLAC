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

# Get Dataset
file = [
    'CICItoCICI',
    #'CICItoUNSW',
    #'UNSWtoUNSW',
    #'UNSWtoCICI',
    #'MergedCICI',
    #'MergedUNSW'
]

data = pd.read_csv('/home/wendyunji/MLAC_steps/ProcessedDataset/'+file[0]+'.csv')
target = data['attack_category']
data = data.drop(labels=['label','attack_category'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=True, stratify=target, random_state=34)

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

# Return Evaluation Metrics
accuracy = pd.DataFrame(columns=['Model','Acc','F1_mi','Recall_mi','Precision_mi','F1_ma','Recall_ma','Precision_ma','F1_we','Recall_we','Precision_we','Execution'])
print('Model\tAcc\tF1_mi\tRecall_mi\tPrecision_mi\tF1_ma\tRecall_ma\tPrecision_ma\tF1_we\tRecall_we\tPrecision_we\tExecution')
cnt = 0
for name, model in models:
    start_time = time.time()
    # Training
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    # Evaluation
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
    # Confusion metrics
    confusion = metrics.confusion_matrix(y_test, prediction)
    # Save
    accuracy.loc[cnt] = [name, acc, f1_mi, f1_ma, f1_we, recall_mi, recall_ma, recall_we, precision_mi, precision_ma, precision_we, delta]
    print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f} secs'.format(name, acc, f1_mi, recall_mi, precision_mi, f1_ma, recall_ma, precision_ma, f1_we, recall_we, precision_we, delta))
    cnt += 1
    matrix = open('/home/irteam/wendyunji-dcloud-dir/wendyunji/MLAC/Classification/Evaluation/Multi/Matrix/'+file[0]+'_'+name+'.txt','w')
    matrix.write(str(confusion))            
accuracy = accuracy.round(3)
accuracy.to_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/MLAC/Classification/Evaluation/Multi/'+file[0]+'.csv',index=False)