from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def create_pipeline(model):
    imputer=SimpleImputer(strategy='mean')
    scaler=MinMaxScaler()
    pipeline=Pipeline(steps=[('imputer',imputer),('scaler',scaler),('model',model)])
    return pipeline


class Multiclass:
    def __init__(self,n_classes,model):
        self.n_classes=n_classes
        self.model=model
        self.y_pred=[]

    def train_test_split(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True, stratify=target, random_state=34)
        return X_train,X_test,y_train,y_test

    def fit(self,X_train,y_train):
        
        self.model.fit(X_train,y_train)

        # 4개 클래스 분류 진행
        

        # 15개 클래스 분류 진행
