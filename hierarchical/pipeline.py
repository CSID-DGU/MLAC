from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def create_pipeline(model):
    imputer=SimpleImputer(strategy='mean')
    scaler=MinMaxScaler()
    pipeline=Pipeline(steps=[('imputer',imputer),('scaler',scaler),('model',model)])
    return pipeline


