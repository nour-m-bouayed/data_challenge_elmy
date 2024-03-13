binarize = True
remove_trend = False  # Bad parameters (decrease performance)
lag_features = True
input_nan = True

split_ratio = 0.7

save_model = False
model_name = "XGBReg"
parameters = "lagfeatures_removetrend_betterfeatures"

import pandas as pd
from src.models import build_model
from src.processing import process_features, process_target, impute_na
from src.evaluation import evaluate, evaluate_forecaster
from src.metrics import weighted_accuracy

# Inport SVR model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=DataConversionWarning)

print("Loading data...")
x_train = pd.read_csv("data/x_train.csv", index_col="DELIVERY_START", parse_dates=True)
y_train = pd.read_csv("data/y_train.csv", index_col="DELIVERY_START", parse_dates=True)
x_test = pd.read_csv("data/x_test.csv", index_col="DELIVERY_START", parse_dates=True)

# Split data
N_train = int(len(x_train) * split_ratio)
x_train_eval, _, train_indices, _ = process_features(
    x_train,
    x_train,
    remove_trend=remove_trend,
    lag_features=lag_features,
    impute_nan=input_nan,
)
y_train_eval = y_train
y_train_eval, y_indices = process_target(
    y_train_eval, binarize=binarize, impute_nan=input_nan
)

print("Training model...")
# Train model
model = HistGradientBoostingClassifier()
# Perform grid search
parameters = {
    "max_depth": [5, 10, 20, None],
    "learning_rate": [0.1, 0.01, 0.001],
    "max_iter": [100, 200, 300],
    "l2_regularization": [0, 1e-3, 1e-1],
    "max_leaf_nodes": [31, 63, 127, None],
    "min_samples_leaf": [20, 50, 100],
}

grid = GridSearchCV(
    model,
    parameters,
    n_jobs=-1,
    scoring=make_scorer(weighted_accuracy),
    cv=5,
    verbose=3,
)

grid.fit(x_train_eval.iloc[train_indices], y_train_eval.iloc[y_indices].values.flatten())

print(grid.best_params_, grid.best_score_)

# Save results
results = pd.DataFrame(grid.cv_results_)
results.to_csv(f"grid_search_{model_name}.csv")
