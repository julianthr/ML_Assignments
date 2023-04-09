import tensorflow as tf
from tensorflow import keras
import numpy as np


# data import
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train_full = X_train_full / 255.
X_test = X_test / 255.

# validation split
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# set seeds
np.random.seed(123)
tf.random.set_seed(123)

# model construction
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation="relu"),  # convolution layer, kernel_size=[5x5]
    keras.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation="relu"),  # convolution layer, kernel_size=[5x5]
    keras.layers.BatchNormalization(),  # batch normalization layer
    keras.layers.MaxPool2D(pool_size=2),  # Max pooling with pool size [2x2]
    keras.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation="relu"),  # convolution layer, kernel_size=[5x5]
    keras.layers.Conv2D(filters=128, kernel_size=5, padding="same", activation="relu"),  # convolution layer, kernel_size=[5x5]
    keras.layers.BatchNormalization(),  # batch normalization layer
    keras.layers.MaxPool2D(pool_size=2),  # Max pooling with pool size [2x2]
    keras.layers.Conv2D(filters=256, kernel_size=5, padding="same", activation="relu"),  # convolution layer, kernel_size=[5x5]
    keras.layers.Conv2D(filters=512, kernel_size=5, padding="same", activation="relu"),  # convolution layer, kernel_size=[5x5]
    keras.layers.Conv2D(filters=1024, kernel_size=5, padding="same", activation="relu"),  # convolution layer, kernel_size=[5x5]
    keras.layers.MaxPool2D(pool_size=2),  # Max pooling with pool size [2x2]
    keras.layers.Flatten(),  # necessary so that dimensionality of the dense layer fits
    keras.layers.Dropout(rate=0.5),  # dropout layer
    keras.layers.Dense(units=512, activation="softmax"),  # dense layer with 512 units
    keras.layers.Dense(units=256, activation="softmax"),  # dense layer with 256 units
    keras.layers.Dropout(rate=0.5),  # dropout layer
    keras.layers.Dense(units=10, activation="softmax")])  # dense layer with 256 units

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=[X_valid, y_valid])

accuracy = model.evaluate(X_test, y_test)
print(accuracy)
