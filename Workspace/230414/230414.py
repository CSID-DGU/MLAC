import pandas as pd

ConcatedCICI = pd.read_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/Dataset/MLAC-dataset/encoded_ConcatedCICI.csv')
#ConcatedUNSW = pd.read_csv('/home/irteam/wendyunji-dcloud-dir/wendyunji/Dataset/MLAC-dataset/encoded_ConcatedUNSW.csv')

del ConcatedCICI['protocol']
del ConcatedCICI['fwd_psh_flags']
del ConcatedCICI['bwd_psh_flags']
del ConcatedCICI['urg_flag_cnt']
del ConcatedCICI['fwd_byts_b_avg']
del ConcatedCICI['init_fwd_win_byts']
del ConcatedCICI['fwd_seg_size_min']

import classification_util2 as Cutils

files = ['ConcatedCICI']
datas = [ConcatedCICI]

def Cpipeline(file, data):
    X_train, X_test, y_train, y_test = Cutils.TrainTestSplit(data)
    MLmodels = Cutils.getModels()
    Cutils.MultiClassification(file, MLmodels, X_train, X_test, y_train, y_test)

for i in range(len(files)):
    Cpipeline(files[i],datas[i])
