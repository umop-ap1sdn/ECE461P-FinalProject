import pandas as pd

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



starting_data = pd.read_csv("BTCdata.csv")

### interpolate missing values by finding where EPOCH time differences are not 60 ###

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

reversed = fixed.iloc[::-1]

reversed.to_csv("BTCdata_clean.csv", index=False)
