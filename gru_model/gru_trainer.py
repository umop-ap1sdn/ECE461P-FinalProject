import keras
from keras import Sequential
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import layers

import numpy as np
import pandas as pd

# Create model
model = Sequential(name="BitcoinGRU")
model.add(layers.Input(shape=(32, 32), batch_size=32))
model.add(layers.GRU(32, kernel_regularizer=regularizers.L1L2(0.01, 0.01)))
model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.L1L2(0.01, 0.01)))
model.add(layers.Dense(1, activation='tanh', kernel_regularizer=regularizers.L1L2(0.01, 0.01)))

model.compile(
    optimizer=optimizers.Adam(learning_rate=4e-4, beta_1=0.2, beta_2=0.2),
    loss=losses.MeanSquaredError(reduction='sum_over_batch_size')
)

model.summary()