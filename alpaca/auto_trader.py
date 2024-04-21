import requests
import json
import os

import pandas as pd
import numpy as np

from queue import Queue



########################################################
# 
# This section should be used by Everyone
# 
########################################################

def getRecentData():
    url = 'https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60'
    r = requests.get(url)

    array = eval(r.text)
    columns = ['Epoch Time', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']

    return pd.DataFrame(array, columns=columns)

def find_missing(epoch):
    missing = []
    for i in range(len(epoch) - 1):
        if epoch[i] - epoch[i + 1] != 60:
            add = (i, int((epoch[i] - epoch[i + 1]) / 60))
            missing.append(add)

    return missing

def run_interpolation(start, end, steps):
    ret = []
    stepSize = round((end - start) / steps, 2)

    for i in range(1, steps):
        ret.append(start + (stepSize * i))
    return ret

def interpolate(column, missing):
    missingI = 0
    columnI = 0

    new_column = []
    
    while columnI < len(column):
        new_column.append(column[columnI])
        
        if missingI < len(missing) and columnI == missing[missingI][0]:
            interpolated = run_interpolation(column[columnI], column[columnI + 1], missing[missingI][1])
            
            for i in interpolated:
                new_column.append(i)
            
            missingI = missingI + 1
        columnI = columnI + 1
        
            
    return new_column

########################################################
# 
# Call THIS function to get a dataframe of cleaned
# data of the most recent 300 datapoints
# 
########################################################

def basicPrepare():
    starting_data = getRecentData()
    epoch = starting_data["Epoch Time"]
    missing = find_missing(epoch)

    filled_epoch = interpolate(epoch, missing)
    filled_open = interpolate(starting_data["Open Price"], missing)
    filled_high = interpolate(starting_data["High Price"], missing)
    filled_low = interpolate(starting_data["Low Price"], missing)
    filled_close = interpolate(starting_data["Close Price"], missing)
    filled_volume = interpolate(starting_data["Volume"], missing)


    fixed = pd.DataFrame()
    fixed["Epoch Time"] = filled_epoch
    fixed["Open Price"] = filled_open
    fixed["High Price"] = filled_high
    fixed["Low Price"] = filled_low
    fixed["Close Price"] = filled_close
    fixed["Volume"] = filled_volume

    return fixed.iloc[::-1]

########################################################
# 
# Create your own file 'alpaca/keys.csv'
# File should be formatted as:
# 
# public,private
# {YOUR_PUBLIC_KEY},{YOUR_PRIVATE_KEY}
# 
########################################################
keys = pd.read_csv('alpaca/keys.csv')

alpaca_id = keys['public'][0]
alpaca_secret = keys['private'][0]

queue = Queue()

########################################################
# 
# These functions are for performing ALPACA trades
# for ALL instances of your program call load and save
# queue before performing any actions
# 
# Use buy() to purchase $50 worth of bitcoin
# This will automatically be added to a queue
# Use sell() to sell the exact amount of bitcoin
# purchased in the first available queue item
# 
# all instances should begin with calling queue=build_queue
# 
########################################################


# This function is preferably not called by the user at all
def add_buy_to_queue():
    url = "https://paper-api.alpaca.markets/v2/orders?status=closed&symbols=BTC%2FUSD&direction=asc"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": alpaca_id,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    response = requests.get(url, headers=headers)

    data = json.loads(response.text)

    amount = data[0]['filled_qty']

    print("Bought", amount, "bitcoin")

    queue.put(amount)
    

def buy():
    url = "https://paper-api.alpaca.markets/v2/orders"

    payload = {
        "side": "buy",
        "type": "market",
        "time_in_force": "ioc",
        "symbol": "BTC/USD",
        "notional": "50"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "APCA-API-KEY-ID": alpaca_id,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    response = requests.post(url, json=payload, headers=headers)

    print("Status code:", response.status_code)
    if response.status_code == 200:
        add_buy_to_queue()

# Preferably this also is not called by anyone directly, only if the sell call fails
def sell_all():
    print("attempting to sell all remaining bitcoin.")
    url = "https://paper-api.alpaca.markets/v2/positions/BTCUSD?percentage=100"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": alpaca_id,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    response = requests.delete(url, headers=headers)

    if response.status_code == 200:
        print('success, emptying queue')
        while not queue.empty():
            queue.get()

def sell():
    if queue.empty():
        print("Nothing to sell")
        return

    amount = queue.get()
    
    url = "https://paper-api.alpaca.markets/v2/orders"

    payload = {
        "side": "sell",
        "type": "market",
        "time_in_force": "ioc",
        "symbol": "BTC/USD",
        "qty": str(amount)
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "APCA-API-KEY-ID": alpaca_id,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    response = requests.post(url, json=payload, headers=headers)
    print("Status code:", response.status_code)
    if response.status_code == 200:
        print("Sold", amount, "bitcoin")
    elif response.status_code == 403:
        print("Attempted to oversell supply, selling all bitcoin")
        sell_all()
    else:
        queue.put(amount)

def build_queue():
    url = "https://paper-api.alpaca.markets/v2/orders?status=closed&symbols=BTC%2FUSD&direction=asc"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": alpaca_id,
        "APCA-API-SECRET-KEY": alpaca_secret
    }

    response = requests.get(url, headers=headers)

    data = json.loads(response.text)

    temp_queue = Queue()
    
    for order in data:
        if order['side'] == 'sell' and not temp_queue.empty():
            temp_queue.get()
        else:
            temp_queue.put(order['filled_qty'])
        
    return temp_queue

sell_all()
