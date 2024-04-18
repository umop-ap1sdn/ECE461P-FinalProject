# from keras.models import load_model

import requests
import json

import pandas as pd


def getRecentData():
    url = 'https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60'
    r = requests.get(url)

    array = eval(r.text)
    columns = ['Epoch Time', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']

    return pd.DataFrame(array, columns=columns)


