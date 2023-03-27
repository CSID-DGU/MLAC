# Import libraries
import pandas as pd
import numpy as np
import glob
import os

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
#from skimpy import clean_columns

# Set printing options
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
pd.set_option('display.max_rows',None)

"""
Loading csv dataset into DataFrame - get unencoded dataset
Input : filename
Output : dataframe
"""
def getData(file):
    data = pd.read_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/Dataset/MLAC-dataset/original_'+file+'.csv')
    return data

"""
Clean Columns
Input : filename, dataframe
output : dataframe with unnecessary columns removed & cleaned(smaller letter) column names
** if CICI format : 'Unnamed: 0','Flow ID','Timestamp','Src IP', 'Dst IP', 'Src Port', 'Dst Port' removed
** else UNSW format : 'Unnamed: 0','srcip','dstip','sport','dsport','Stime','Ltime' removed
"""
def cleanColumns(file, data):
    data = clean_columns(data)

    # Drop Unnecessary Columns
    if file.endswith('CICI'):
        data.drop(columns=['unnamed_0','flow_id','timestamp','src_ip', 'dst_ip', 'src_port', 'dst_port'], inplace=True)
    else:
        data.drop(columns=['unnamed_0','srcip','dstip','sport','dsport','stime','ltime'], inplace=True)
    
    # Clean column names
    if 'attack_cat' in data.columns:
        data = data.rename(columns={'attack_cat':'attack_category'})

    return data

"""
getXdata - Drop label & attack_category
input : file name & dataframe
output : x_data frame
"""
def getX(file, data):
    x_data = data.drop(['label','attack_category'], axis=1)
    return x_data

"""
devide features into numeric & categorical
input : x_data frame
output : numeric & categorical features name list
"""
def devideFeatures(x_data):
    numeric_features = list(x_data.select_dtypes(exclude=[object]).columns)
    categorical_features = list(x_data.select_dtypes(exclude=["number"]).columns)
    return numeric_features, categorical_features

"""
Print numeric & categorical features
input : dataframe, numeric & categorical feature
output : print numeric & categorical feature list
"""
def printFeatures(data, numeric_features, categorical_features):
    print('######### All Columns #########')
    print('Total Column lenth :',len(data.columns))
    print(data.columns)
    print('######### Numeric Columns #########')
    print('Numeric Column lenth :',len(numeric_features))
    print(numeric_features)    
    print('######### Categorical Columns #########')
    print('Categorical Column lenth :',len(categorical_features))
    print(categorical_features)

"""
XEcoding - Quantile Transform for numeric features & One-Hot Encoding for categorical features
input : x_data frame, numeric & categorical feature list
output : Encoded x_data frame
"""
def xEncoding(x_data, numeric_features, categorical_features):
    # Scale Numeric Data by using QuantileTransformer
    numeric_data = x_data.drop(categorical_features, axis=1)
    numeric_data = pd.DataFrame(QuantileTransformer().fit_transform(numeric_data), columns=numeric_features)

    # Scale Categorical Data by using QuantileTransformer
    categorical_data = x_data.drop(numeric_features, axis=1)
    if len(categorical_features) != 0:
        categorical_data = pd.get_dummies(categorical_data)

    # Concat X data
    x_data = pd.concat([numeric_data, categorical_data],axis=1)
    return x_data, numeric_data, categorical_data

"""
yEncoding - Label Encoding for attack category
input : full data frame
output : Encoded y_data frame
"""
def yEncoding(data):
    le = LabelEncoder()
    data.loc[data['attack_category'] == 'Normal', 'attack_category'] = '0'
    data.loc[data['attack_category'] == 'BENIGN', 'attack_category'] = '0'
    y_data = data['attack_category']
    y_data = pd.DataFrame(le.fit_transform(y_data), columns=["attack_category"])
    y_data['label'] = data['label']
    return y_data

"""
return Encoded data frame
input : file name, encoded x & y dataframe
output : save and return full encoded data frame
"""
def returnData(file, x_data, y_data):
    encoded = pd.concat([x_data,y_data],axis=1)
    encoded.dropna(axis=0, inplace=True)
    encoded.to_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/Dataset/MLAC-dataset/encoded_'+file+'.csv', index=False)
    return encoded