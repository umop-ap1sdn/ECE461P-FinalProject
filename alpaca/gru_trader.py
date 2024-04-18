from keras.models import load_model
import tensorflow as tf

import auto_trader

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

########################################################
# 
# This section is used specifically for the GRU
# Prepares data based on the specifications for the GRU
# 
########################################################

def lagging_series(column, size):
    lagging = []
    target_column = []
    for i in range(1, size + 1):
        new_column = []
        for j in range(size, len(column)):
            if i == 1:
                target_column.append(column[j])
            new_column.append(column[j - i])
        lagging.append(new_column)

    return lagging, target_column

def difference(column):
    new_column = []

    for i in range(1, len(column)):
        new_column.append(round(column[i] - column[i - 1], 3))
    
    return new_column

def percent_difference(column):
    new_column = []

    for i in range(1, len(column)):
        new_column.append(round((column[i] - column[i - 1]) / column[i - 1], 20))
    
    return new_column


def binary_prep(target_price):
    binary = []
    price = 0
    for i in range(10):
        price += target_price[i]
    
    if price > 0: binary.append(1)
    else: binary.append(0)

    for i in range(10, len(target_price)):
        price -= target_price[i - 10]
        price += target_price[i]
        if price > 0: binary.append(1)
        else: binary.append(0)
    
    return binary

def gru_prepare(clean, price_scaler, volume_scaler):
    diff_price = difference(clean['Open Price'])
    diff_vol = difference(clean['Volume'])

    percent_price = percent_difference(clean['Open Price'])
    percent_vol = percent_difference(clean['Volume'])

    ### Use Percent Price, Difference Volume

    price = price_scaler.transform(np.array(percent_price).reshape(-1, 1)).flatten().tolist()
    volume = volume_scaler.fit_transform(np.array(diff_vol).reshape(-1, 1)).flatten().tolist()

    price = [round(x, 4) for x in price]
    volume = [round(x, 4) for x in volume]


    lagging_size = 32
    lagging_price, target_price = lagging_series(price, lagging_size)
    lagging_volume, target_volume = lagging_series(volume, lagging_size)

    binary = binary_prep(target_price)

    df = pd.DataFrame()
    extra = pd.DataFrame()

    df['binary'] = binary

    for i in range(lagging_size):
        df['p[t-' + str(i) + ']'] = lagging_price[i][:-9]
        extra['p[t-' + str(i) + ']'] = lagging_price[i][-9:]

    for i in range(lagging_size):
        df['v[t-' + str(i) + ']'] = lagging_volume[i][:-9]
        extra['v[t-' + str(i) + ']'] = lagging_volume[i][-9:]

    return df, extra

def gru_xy(dataset):
    train_cols = np.arange(1, 33)
    train_cols = np.append(train_cols, np.arange(33, 65))

    X = dataset.iloc[:-1, train_cols]
    X['bin'] = dataset.iloc[:-1, 0]
    y = dataset.loc[1:, 'binary']

    X = X.to_numpy()
    y = y.to_numpy()

    return X, y

def fit_scalers():

    price_scaler = StandardScaler()
    volume_scaler = StandardScaler()

    ### Fit scalers on original data
    pre_cleaned = pd.read_csv("BTCdata_clean.csv")

    percent_price_orig = percent_difference(pre_cleaned['Open Price'])
    diff_vol_orig = difference(pre_cleaned['Volume'])

    price_scaler.fit(np.array(percent_price_orig).reshape(-1, 1))
    volume_scaler.fit(np.array(diff_vol_orig).reshape(-1, 1))

    return price_scaler, volume_scaler

def predict_primary(model, dataset):
    y_pred = model.predict(np.reshape(dataset, (-1, 1, dataset.shape[1])))
    return y_pred

def extrapolate(model, initial_pred, extra):
    print(initial_pred)
    y_pred = [initial_pred]
    y_prev = initial_pred
    for i in range(extra.shape[0]):
        input = np.append(extra[i, :], [y_prev], axis=0)
        input = np.reshape(input, (-1, 1, extra.shape[1] + 1))
        y_prev = model.predict(input)[0, 0, 0]
        y_pred.append(y_prev)
       
    
    return y_pred

def run_prediction():
    # Do this first as it is a long running function that does not depend on the newest data
    price_scaler, volume_scaler = fit_scalers()
    # Pre-trained model
    model = load_model('gru_bitcoin.keras')

    # Data is collected here, timing is important after this moment
    data = auto_trader.basicPrepare()

    # Create dataset, prepared in the same way as the initial
    # Extra dataset refers to the left over values that get cut off because of the forward binary values
    data, extra = gru_prepare(data, price_scaler=price_scaler, volume_scaler=volume_scaler)
    X, y = gru_xy(data)

    # Create base predictions of the dataset
    y_pred = predict_primary(model, X)
    # Create extrapolated predictions from the extra data, and most previously predicted value
    y_pred = extrapolate(model, y_pred[-1, 0, 0], extra.to_numpy())

    print(y_pred)
    return y_pred[-1]

run_prediction()