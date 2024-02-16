from sklearn.preprocessing import StandardScaler
import pandas as pd

def process_features(x_train,x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)
    x_train_scaled.drop(columns=['predicted_spot_price'], inplace=True)
    x_test_scaled.drop(columns=['predicted_spot_price'], inplace=True)

    x_train_scaled.ffill(inplace=True)
    x_test_scaled.ffill(inplace=True)

    return x_train_scaled, x_test_scaled