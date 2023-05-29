import classification_util
import pandas as pd

# 데이터 및 모델 임포트
files={
    'CICI':'/home/irteam/dcloud-global-dir/MLAC/new_data/CICI.csv',
    'UNSW':'/home/irteam/dcloud-global-dir/MLAC/new_data/UNSW.csv'
}

CICI_data = pd.read_csv(files['CICI'])
UNSW_data = pd.read_csv(files['UNSW'])

models = classification_util.getModels()

# 훈련 및 평가
CICI_mb_X_train, CICI_mb_X_test, CICI_mb_y_train, CICI_mb_y_test =classification_util.TrainTestSplit('B', CICI_data)
classification_util.BinaryClassification(dtype = 'CICI', file='CICI_mb', models=models, X_train = CICI_mb_X_train, X_test = CICI_mb_X_test, y_train = CICI_mb_y_train, y_test = CICI_mb_y_test)

UNSW_mb_X_train, UNSW_mb_X_test, UNSW_mb_y_train, UNSW_mb_y_test =classification_util.TrainTestSplit('B', UNSW_data)
classification_util.BinaryClassification(file='UNSW_mb', models=models, X_train = UNSW_mb_X_train, X_test = UNSW_mb_X_test, y_train = UNSW_mb_y_train, y_test = UNSW_mb_y_test)