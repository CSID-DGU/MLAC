import pandas as pd

#ConcatedUNSW = pd.read_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/Dataset/MLAC-dataset/encoded_ConcatedUNSW.csv')
ConcatedUNSW = pd.read_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/Dataset/MLAC-dataset/encoded_ConcatedUNSW.csv')

del ConcatedUNSW['protocol']
del ConcatedUNSW['fwd_psh_flags']
del ConcatedUNSW['bwd_psh_flags']
del ConcatedUNSW['urg_flag_cnt']
del ConcatedUNSW['fwd_byts_b_avg']
del ConcatedUNSW['init_fwd_win_byts']
del ConcatedUNSW['fwd_seg_size_min']

import classification_util2 as Cutils

files = ['ConcatedUNSW']
datas = [ConcatedUNSW]

def Cpipeline(file, data):
    X_train, X_test, y_train, y_test = Cutils.TrainTestSplit(data)
    MLmodels = Cutils.getModels()
    Cutils.MultiClassification(file, MLmodels, X_train, X_test, y_train, y_test)

for i in range(len(files)):
    Cpipeline(files[i],datas[i])
