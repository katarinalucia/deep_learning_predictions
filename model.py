"""
def f(x):
    y = (2 * x) - 1
    return y
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras


xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


model = tf.keras.Sequential([
   keras.layers.Dense(units=4, input_shape=[1]),
   keras.layers.Dense(units=8),
   keras.layers.Dense(units=1),
   ])
model.summary()

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=100)

print(model.predict([10.0]))

model.save('model')