import auto_trader
import time
import dataset_creator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

auto_trader.buy()
auto_trader.queue = auto_trader.build_queue()
from keras.models import load_model, save_model

model = load_model('mlp0.53.h5')

model.summary()

for i in range(72):
    data = auto_trader.basicPrepare()
    diff_price = dataset_creator.difference(data['Open Price'])
    diff_vol = dataset_creator.difference(data['Volume'])

    percent_price = dataset_creator.percent_difference(data['Open Price'])
    percent_vol = dataset_creator.percent_difference(data['Volume'])

    ### Use Percent Price, Difference Volume

    scaler = StandardScaler()
    price = scaler.fit_transform(np.array(percent_price).reshape(-1, 1)).flatten().tolist()
    volume = scaler.fit_transform(np.array(diff_vol).reshape(-1, 1)).flatten().tolist()

    price = [round(x, 4) for x in price]
    volume = [round(x, 4) for x in volume]


    lagging_size = 32
    lagging_price, target_price = dataset_creator.lagging_series(price, lagging_size)
    lagging_volume, target_volume = dataset_creator.lagging_series(volume, lagging_size)


    df = pd.DataFrame()


    for i in range(lagging_size):
        df['p[t-' + str(i) + ']'] = lagging_price[i][:-9]

    for i in range(lagging_size):
        df['v[t-' + str(i) + ']'] = lagging_volume[i][:-9]

    proba = model.predict(df.iloc[-1].values.reshape(1,-1))

    current_log = []

    BUY_THRESHOLD = 0.5
    SELL_THRESHOLD = 0.5


    # '''
    if proba >= BUY_THRESHOLD:
        print("Buying Bitcoin")
        current_log.append('bought')
        auto_trader.buy()
    elif proba <= SELL_THRESHOLD:
        print("Selling Bitcoin")
        current_log.append('sold')
        auto_trader.sell()
    else:
        current_log.append('neither')
        print('Uncertainty too high to trade')

    # '''

    time.sleep(600)