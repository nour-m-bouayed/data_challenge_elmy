from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def build_matrix_harmonic_reg(freqs, time_array):
    A_matrix = np.array([(np.cos(2 * np.pi * fi * time_array), np.sin(2 * np.pi * fi * time_array)) for fi in freqs])
    A_matrix = A_matrix.reshape(len(freqs)*2,-1)
    A_matrix = np.vstack((A_matrix, np.array([np.ones(time_array.size)])))
    A_matrix = A_matrix.T
    return A_matrix

def get_largest_local_max(signal1D, n_largest = 3, order = 1):
    """Return the largest local max and the associated index in a tuple.

    This function uses `order` points on each side to use for the comparison.
    """
    all_local_max_indexes = argrelmax(signal1D, order=order)[0]
    all_local_max = np.take(signal1D, all_local_max_indexes)
    largest_local_max_indexes = all_local_max_indexes[all_local_max.argsort()[::-1]][
        :n_largest
    ]

    return (
        np.take(signal1D, largest_local_max_indexes),
        largest_local_max_indexes,
    )

# function to partition a time series into intervals of NaNs and intervals of non-NaNs
def partition_na(column):
    '''
    Input :
        column (pd.Series with a datetime-like index)
    Returns : 
        null_patches (list) : list of intervals (list of consecutive points)
        
    '''
    null_indices = column[column.isnull()].index
    null_patches = []
    patch = [null_indices[0]]
    assert len(null_indices)>0
    for i in range(1,len(null_indices)):
        if null_indices[i] != null_indices[i-1]+pd.Timedelta('1h'):
            null_patches.append(patch)
            patch = [null_indices[i]]
        else:
            patch.append(null_indices[i])
    null_patches.append(patch)
    non_null_patches = []
    for i in range(len(null_patches)+1):
        # non_null interval situated directly on the left of the current null interval
        if i==0:
            if null_patches[i][0]==column.index[0]: #column starts with a NaN
                nan_first = True
                continue
            else:
                nan_first = False
                l1 = column.index[0]
                l2 = null_patches[i][-1] - pd.Timedelta('1h')           
        elif i==len(null_patches):
            if null_patches[i-1][-1]==column.index[-1]: #column ends with a NaN
                nan_last = True
                continue
            else:
                nan_last = False
                l1 = null_patches[i-1][-1] + pd.Timedelta('1h')
                l2 = column.index[-1]
        else:
            l1 = null_patches[i-1][-1] + pd.Timedelta('1h')
            l2 = null_patches[i][0] - pd.Timedelta('1h')
        if len(list((column[l1:l2])))>10:
            non_null_patches.append(list((column[l1:l2]).index))
        
    return null_indices, null_patches, non_null_patches, nan_first, nan_last

def relu(x):
    if x<0 :
        return 0
    else:
        return x
    
def get_hours_from_date(time_index, date):
    # Convert the datetime column into timedelta relative to the given date
    time_delta = time_index - date
    # Convert timedelta to hours (as integers)
    hours = time_delta.total_seconds() // 3600  # 3600 seconds in an hour
    # Convert Hours column to a list of integers
    hours_list = np.array(hours.astype(int).tolist())
    return hours_list

# function to perform missing values imputation
def impute_na(df, plot=False, trend_degree=9, seasonality_nb_freqs=4):
    '''
    Returns :
        df_imputed (DataFrame) : df resampled at '1h' intervals, without NaNs
        original_df_indices (ndarray) : indices of rows in `df_imputed` mapping to original df rows
    '''
    df_temp = df.copy()
    df_temp['original'] = 1
    df_temp = df_temp.resample('1h').mean()
    original_df_indices = np.where(df_temp['original'] == 1)

    na_columns = df_temp.columns[df_temp.isna().sum()>0]
    cols_to_drop = ['original']
    for column in na_columns:
        print('COLUMN : ', column)
        try:
            # Get trend of column signal
            signal = df_temp[column].interpolate()
            time_array = np.arange(len(signal.index))
            coeffs = np.polyfit(time_array, signal, trend_degree)
            approx_poly_trend = np.polyval(coeffs, time_array)
            df_temp[f'trend_{column}'] = approx_poly_trend
            cols_to_drop.append(f'trend_{column}')

            null_indices, null_patches, non_null_patches, _, _ = partition_na(df_temp[column])
            for i in range(len(null_patches)):
                # Get local seasonality (from signal on the left or on the right )
                if (len(null_patches) > len(non_null_patches)) and i==len(null_patches)-1:
                    non_null_interval = non_null_patches[i-1]
                else:
                    non_null_interval = non_null_patches[i]

                signal = (df_temp[column][non_null_interval] - df_temp[f'trend_{column}'][non_null_interval]).interpolate()
                time_array = np.arange(signal.size)
                fft_values = np.fft.fft(signal) 
                frequencies = np.fft.fftfreq(len(signal), 3600)
                # positive_freqs = frequencies[:len(frequencies)//2]
                magnitude_spectrum = np.abs(fft_values[:len(fft_values)//2])
                max_freq, max_freq_idx =get_largest_local_max(magnitude_spectrum, n_largest=seasonality_nb_freqs)
                freqs = frequencies[max_freq_idx]*3600 #in H^{-1}
                A_matrix = build_matrix_harmonic_reg(freqs, time_array)
                reg_coeffs = np.linalg.lstsq(A_matrix, signal, rcond=None)[0]
                hours_list = get_hours_from_date(df_temp.loc[null_patches[i],:].index, signal.index[0])
                A_matrix_na = build_matrix_harmonic_reg(freqs, hours_list)
                df_temp.loc[null_patches[i], column] = df_temp.loc[null_patches[i], f'trend_{column}'] + A_matrix_na @ reg_coeffs
                df_temp.loc[null_patches[i], column] = df_temp.loc[null_patches[i], column].apply(relu)
            if plot:
                plt.figure(figsize=(30,5))
                plt.plot(df_temp[column])
                for patch in null_patches:
                    plt.plot(df_temp.loc[patch,column], color='red')
                plt.title(f'{column}')
        except:
            print(f"Could not decompose {column} into trend/seasonality, doing simple interpolation")
            null_indices, null_patches, non_null_patches, _, _ = partition_na(df_temp[column])
            df_temp[column] = df_temp[column].fillna(method = 'ffill')

            if plot:
                plt.figure(figsize=(30,5))
                plt.plot(df_temp[column])
                for patch in null_patches:
                    plt.plot(df_temp.loc[patch,column], color='red')
                plt.title(f'{column}')
    df_temp.drop(cols_to_drop, axis=1, inplace=True)
    
    return df_temp, original_df_indices

def process_features(x_train, x_test, remove_trend=False, lag_features=False):

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
