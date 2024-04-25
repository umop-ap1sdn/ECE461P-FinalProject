import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

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


def binaryfunc(target_price):
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

clean = pd.read_csv("BTCdata_clean.csv")
diff_price = difference(clean['Open Price'])
diff_vol = difference(clean['Volume'])

percent_price = percent_difference(clean['Open Price'])
percent_vol = percent_difference(clean['Volume'])

### Use Percent Price, Difference Volume

scaler = StandardScaler()
price = scaler.fit_transform(np.array(percent_price).reshape(-1, 1)).flatten().tolist()
volume = scaler.fit_transform(np.array(diff_vol).reshape(-1, 1)).flatten().tolist()

price = [round(x, 4) for x in price]
volume = [round(x, 4) for x in volume]


lagging_size = 32
lagging_price, target_price = lagging_series(price, lagging_size)
lagging_volume, target_volume = lagging_series(volume, lagging_size)

# binary = binary(target_price)

# '''
# df = pd.DataFrame()

# df['Target_Price'] = target_price
# df['Target_Volume'] = target_volume

# for i in range(lagging_size):
#     df['p[t-' + str(i) + ']'] = lagging_price[i]

# for i in range(lagging_size):
#     df['v[t-' + str(i) + ']'] = lagging_volume[i]

# df.to_csv("BTCdata_final_round.csv", index=False)
# '''

# # fig, ax = plt.subplots()
# # bins = np.linspace(-20, 20, 1000)
# # ax.hist(diff_vol, bins=bins)
# # plt.show()

# df = pd.DataFrame()

# df['binary'] = binary

# for i in range(lagging_size):
#     df['p[t-' + str(i) + ']'] = lagging_price[i][:-9]

# for i in range(lagging_size):
#     df['v[t-' + str(i) + ']'] = lagging_volume[i][:-9]

# df.to_csv("BTCdata_final_binary.csv", index=False)
