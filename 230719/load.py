import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
import os
import matplotlib.pyplot as plt

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

saved_path='/home/irteam/dcloud-global-dir/MLAC/saved_models/230620/savedmodels'
confusion_path='/home/irteam/junghye-dcloud-dir/MLAC/230719/result/CICI/confusion'
out_txt_path='/home/irteam/junghye-dcloud-dir/MLAC/230719/result/CICI/L1_Output.txt'
out_csv_path='/home/irteam/junghye-dcloud-dir/MLAC/230719/result/CICI/ver1.csv'
progressLog_path='/home/irteam/junghye-dcloud-dir/MLAC/230719/result/CICI/L1_progressLog.txt'


def test_result(model:str,test,pred) ->list:

    acc=accuracy_score(test,pred)
    f1=f1_score(test,pred,average='weighted')
    recall=recall_score(test,pred,average='weighted')
    precision=precision_score(test,pred,average='weighted')

    result=[acc,f1,recall,precision]
    result=[round(num,3) for num in result]

    print(f'{model} result , acc:{result[0]}, f1:{result[1]},recall:{result[2]},precision:{result[3]}')
    return(result)


def plot_confusion_matrix(con_mat,labels,title:str,cmap=plt.cm.get_cmap('Blues'),normalize=False):
    plt.figure(figsize=(20,15))
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
    plt.savefig(confusion_path+'/'+title+'.png',facecolor='#eeeeee')
    plt.clf()
