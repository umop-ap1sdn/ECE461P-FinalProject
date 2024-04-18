import keras
from keras import Sequential
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import layers
from keras.layers import Activation
from keras import backend as K
import tensorflow as tf

import numpy as np
import pandas as pd

### Load Data ###
dataset = pd.read_csv("BTCdata_final_round.csv")
print("\n\n\n\n")
print(dataset.head())

X = dataset.loc[:, 'p[t-0]':]
y = dataset.loc[:, 'Target_Price':'Target_Volume']

print(X.shape)
print(y.shape)

train_size = int(0.8 * X.shape[0])

X = X.to_numpy()
y = y.to_numpy()

X_train = X[:train_size, :]
X_test = X[train_size:, :]
y_train = y[:train_size, :]
y_test = y[train_size:, :]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


### Create Model ### 
# @tf.function
# def cube_root_activation(x):
#     return K.sigmoid(x)


batch_size = 64
input_size = 64
output_size = 2

l1 = 0.00
l2 = 0.00

tf.random.set_seed(42)

### Build Model from ground-up ###
model = Sequential(name="BitcoinGRU")
model.add(layers.Input(shape=(None, input_size)))
model.add(layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.L1L2(l1, l2)))
model.add(layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.L1L2(l1, l2)))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L1L2(l1, l2)))
model.add(layers.Dense(output_size, activation=lambda x: x ** 1./3., kernel_regularizer=regularizers.L1L2(l1, l2)))

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
    loss='mse',
    metrics=['mae']
)

model.summary()

print("Working")

X_train = np.reshape(X_train, (-1, 1, 64))
X_test = np.reshape(X_test, (-1, 1, 64))

y_train = np.reshape(y_train, (-1, 1, 2))
y_test = np.reshape(y_test, (-1, 1, 2))


model.fit(x=X_train, y=y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test))
test_loss = model.evaluate(X_test, y_test)

print("Results:", test_loss)