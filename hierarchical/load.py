from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np

def create_pipeline(model):
    imputer=SimpleImputer(strategy='mean')
    scaler=MinMaxScaler()
    pipeline=Pipeline(steps=[('imputer',imputer),('scaler',scaler),('model',model)])
    return pipeline


def test_result(model:str,test,pred) ->list:
    
    acc=accuracy_score(test,pred)
    f1=f1_score(test,pred,average='weighted')
    recall=recall_score(test,pred,average='weighted')
    precision=precision_score(test,pred,average='weighted')
    #confusion=metrics.confusion_matrix(test,pred)
    #plot_confusion_matrix(confusion,labels=list(set(target)),title=model)
    print(f'{model} result , acc:{acc}, f1:{f1},recall:{recall},precision:{precision}')
    return([acc,f1,recall,precision])
    

def plot_confusion_matrix(con_mat,labels,title:str,cmap=plt.cm.get_cmap('Blues'),normalize=False,*args,confusion_path):
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