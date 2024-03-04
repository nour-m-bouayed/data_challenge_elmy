from xgboost import XGBRegressor, XGBClassifier

def build_model(type='regression', lag_features=False):
    
    if type == 'regression':
        parent = XGBRegressor
    elif type == 'classification':
        parent = XGBClassifier

    class MyModel(parent):
        def predict(self, X):
            return super().predict(X)
        
    return MyModel(n_estimators=100, learning_rate=0.1, n_jobs=-1, alpha=0.1, reg_lambda=0.1, max_depth=3,)
        
