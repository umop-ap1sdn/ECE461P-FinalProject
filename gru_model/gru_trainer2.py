import keras
from keras import Sequential
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import layers
from keras.layers import Activation
from keras import backend as K
import tensorflow as tf

from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

### Load Data ###
dataset = pd.read_csv("BTCdata_final_binary.csv")
print("\n\n\n\n")
print(dataset.head())

train_cols = np.arange(1, 33)
train_cols = np.append(train_cols, np.arange(33, 65))

X = dataset.iloc[:-1, np.arange(1,33)]
X['bin'] = dataset.iloc[:-1, 0]
y = dataset.loc[1:, 'binary']

print(X.shape)
print(y.shape)

train_size = int(0.8 * X.shape[0])

X = X.to_numpy()
y = y.to_numpy()

X_train = X[:train_size, :]
X_test = X[train_size:, :]
y_train = np.reshape(y[:train_size], (-1, train_size))
y_test = np.reshape(y[train_size:], (-1, len(y) - train_size))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


### Create Model ### 
# @tf.function
# def cube_root_activation(x):
#     return K.sigmoid(x)


batch_size = 64
input_size = X_train.shape[1]
output_size = 1

l1 = 0.00
l2 = 0.00

dropout = 0.0

tf.random.set_seed(42)

### Build Model from ground-up ###
model = Sequential(name="BitcoinGRU")
model.add(layers.Input(shape=(None, input_size), batch_size=batch_size))
model.add(layers.GRU(100, activation='tanh', kernel_regularizer=regularizers.L1L2(l1, l2), return_sequences=True, dropout=dropout))
# model.add(layers.GRU(128, activation='tanh', kernel_regularizer=regularizers.L1L2(l2, l1), return_sequences=False, dropout=dropout))
model.add(layers.Dense(60, activation='relu', kernel_regularizer=regularizers.L1L2(l1, l2)))
model.add(layers.Dense(output_size, activation='sigmoid', kernel_regularizer=regularizers.L1L2(l1, l2)))

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
)

model.summary()

print("Working")

X_train = np.reshape(X_train, (-1, 1, input_size))
X_test = np.reshape(X_test, (-1, 1, input_size))

y_train = np.reshape(y_train, (-1, 1, output_size))
y_test = np.reshape(y_test, (-1, 1, output_size))


model.fit(x=X_train, y=y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test))
test_loss = model.evaluate(X_test, y_test)

print("Results:", test_loss)

y_train_pred = np.reshape(model.predict(X_train), (-1, 1))
y_test_pred = np.reshape(model.predict(X_test), (-1, 1))

y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

print(y_test_pred)
print(y_test_pred.mean(), y_test_pred.std())
print("Train Score:", roc_auc_score(y_train, y_train_pred))
print("Test Score:", roc_auc_score(y_test, y_test_pred))