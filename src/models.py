from xgboost import XGBRegressor

MyModel = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=0,
    max_depth=3,
    colsample_bytree=0.7,
    subsample=0.7,
    reg_alpha=0.5,
    reg_lambda=0.5,
    tree_method="gpu_hist",
)
