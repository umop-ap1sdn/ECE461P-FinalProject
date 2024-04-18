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

def gru_prepare(clean):
    diff_price = difference(clean['Open Price'])
    diff_vol = difference(clean['Volume'])

    percent_price = percent_difference(clean['Open Price'])
    percent_vol = percent_difference(clean['Volume'])

    ### Use Percent Price, Difference Volume

    price_scaler = StandardScaler()
    volume_scaler = StandardScaler()

    ### Fit scalers on original data
    pre_cleaned = pd.read_csv("BTCdata_clean.csv")

    percent_price_orig = percent_difference(pre_cleaned['Open Price'])
    diff_vol_orig = difference(pre_cleaned['Volume'])

    price_scaler.fit(np.array(percent_price_orig).reshape(-1, 1))
    volume_scaler.fit(np.array(diff_vol_orig).reshape(-1, 1))

    price = price_scaler.transform(np.array(percent_price).reshape(-1, 1)).flatten().tolist()
    volume = volume_scaler.fit_transform(np.array(diff_vol).reshape(-1, 1)).flatten().tolist()

    price = [round(x, 4) for x in price]
    volume = [round(x, 4) for x in volume]


    lagging_size = 32
    lagging_price, target_price = lagging_series(price, lagging_size)
    lagging_volume, target_volume = lagging_series(volume, lagging_size)

    binary = binary_prep(target_price)

    df = pd.DataFrame()

    df['binary'] = binary

    for i in range(lagging_size):
        df['p[t-' + str(i) + ']'] = lagging_price[i][:-9]

    for i in range(lagging_size):
        df['v[t-' + str(i) + ']'] = lagging_volume[i][:-9]

    return df

def gru_xy(dataset):
    train_cols = np.arange(1, 33)
    train_cols = np.append(train_cols, np.arange(33, 65))

    X = dataset.iloc[:-1, train_cols]
    X['bin'] = dataset.iloc[:-1, 0]
    y = dataset.loc[1:, 'binary']

    X = X.to_numpy()
    y = y.to_numpy()

    return X, y

data = auto_trader.basicPrepare()
data = gru_prepare(data)
X, y = gru_xy(data)

print(data)