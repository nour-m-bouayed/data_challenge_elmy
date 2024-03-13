from xgboost import XGBRegressor, XGBClassifier
from darts.models import RNNModel, LinearRegressionModel

def build_model(type='regression', lag_features=False):
    
    if type == 'regression':
        parent = XGBRegressor
    elif type == 'classification':
        parent = XGBClassifier
    elif type == 'forecaster':
        return build_forecaster()

    class MyModel(parent):
        def predict(self, X):
            if type == 'classification':
                return super().predict(X) *2 - 1
            else:
                return super().predict(X)
        
    return MyModel(n_estimators=100, learning_rate=0.1, n_jobs=-1, alpha=0.1, reg_lambda=0.1, max_depth=3,)

def build_forecaster():
    return LinearRegressionModel( lags_future_covariates=(0,1), )
            
