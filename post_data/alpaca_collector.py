import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import matplotlib.pyplot as plt

keys = pd.read_csv('alpaca/keys.csv')

alpaca_id = keys['public'][0]
alpaca_secret = keys['private'][0]

def get_history():
    url = "https://paper-api.alpaca.markets/v2/account/portfolio/history?timeframe=5Min&intraday_reporting=continuous&start=2024-04-20T00%3A00%3A00-05%3A00&pnl_reset=no_reset&end=2024-04-21T00%3A50%3A00-05%3A00"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": alpaca_id,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    response = requests.get(url, headers=headers)
    data = json.loads(response.text)
    
    return data['equity']

def rfc3339_to_unix(rfc3339_timestamp):
    dt = datetime.fromisoformat(rfc3339_timestamp.replace('Z', '+00:00'))
    return int(dt.timestamp())

def get_actual():
    df = pd.read_csv('post_data/BTCdata.csv')
    df = df.loc[:, "Epoch Time": "Volume"]
    return df.iloc[::-1]

def get_orders():
    url = "https://paper-api.alpaca.markets/v2/orders?status=all&symbols=BTC%2FUSD&limit=200"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": alpaca_id,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    response = requests.get(url, headers=headers)
    data = json.loads(response.text)

    buys = pd.DataFrame()
    sells = pd.DataFrame()

    buy_unix = []
    sell_unix = []

    buy_price = []
    sell_price = []

    for order in data:
        if order['side'] == 'buy':
            buy_unix.append(rfc3339_to_unix(order['filled_at']))
            buy_price.append(float(order['filled_avg_price']))
        else:
            sell_unix.append(rfc3339_to_unix(order['filled_at']))
            sell_price.append(float(order['filled_avg_price']))
    
    buys['time'] = buy_unix
    buys['price'] = buy_price

    sells['time'] = sell_unix
    sells['price'] = sell_price

    return buys, sells


'''
equity = np.array(get_history()) - 100000
print(equity)
df = pd.DataFrame()
df['profit'] = equity
df.to_csv('gru_profit.csv', index=False)
'''

buys, sells = get_orders()
print('Buy Average:', buys['price'].sum() / len(buys))
print('Sell Average:', sells['price'][:-1].sum() / (len(sells) - 1))

# '''
df = get_actual()

initial = df.iloc[0, 0]

fig, ax = plt.subplots()
ax.plot(np.array(df['Epoch Time']) - initial, np.array(df['High Price']) + 125)
ax.scatter(np.array(buys['time']) - initial, buys['price'], color='red')
ax.scatter(np.array(sells['time']) - initial, sells['price'], color='green')

plt.show()

# '''

