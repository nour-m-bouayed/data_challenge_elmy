from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def process_features(x_train, x_test,remove_trend=False, lag_features=False):

    x_train_processed = x_train.copy()
    x_test_processed = x_test.copy()

    # Compute the rolling mean of each feature and subtract it from the original feature
    if remove_trend:
        x_train_trend = x_train_processed.rolling(window=24).mean()
        x_test_trend = x_test_processed.rolling(window=24).mean()
        x_train_processed = x_train_processed - x_train_trend
        x_test_processed = x_test_processed - x_test_trend

    scaler = StandardScaler()
    x_train_processed = scaler.fit_transform(x_train_processed)
    x_test_processed = scaler.transform(x_test_processed)

    x_train_processed = pd.DataFrame(x_train_processed, index=x_train.index, columns=x_train.columns)
    x_test_processed = pd.DataFrame(x_test_processed, index=x_test.index, columns=x_test.columns)

    x_train_processed.drop(columns=["predicted_spot_price"], inplace=True)
    x_test_processed.drop(columns=["predicted_spot_price"], inplace=True)

    x_train_processed.ffill(inplace=True)
    x_test_processed.ffill(inplace=True)

    # Augment features with lagged values
    if lag_features:
        column_names = x_train_processed.columns

        x_train_processed = pd.concat({column + "_lag" + str(i): x_train_processed[column].shift(i) for column in column_names for i in range(-12, 12)}, axis=1)
        x_test_processed = pd.concat({column + "_lag" + str(i): x_test_processed[column].shift(i) for column in column_names for i in range(-12, 12)}, axis=1)

    # Add categorical calendar features 
    train_date= pd.to_datetime(x_train_processed.index,utc=True)
    test_date = pd.to_datetime(x_test_processed.index,utc=True)

    x_train_processed["month"] = train_date.month
    x_test_processed["month"] = test_date.month

    x_train_processed["hour"] = train_date.hour
    x_test_processed["hour"] = test_date.hour

    x_train_processed["weekday"] = train_date.weekday
    x_test_processed["weekday"] = test_date.weekday

    x_train_processed["dayofyear"] = train_date.dayofyear
    x_test_processed["dayofyear"] = test_date.dayofyear


    return x_train_processed, x_test_processed




def process_target(y_train, binarize=False):
    if binarize:
        y_train = (y_train >= 0).astype(int)
    return y_train
