# This question is about implementing a simple ANN for a Regression Task
# Use Keras
# Don't do any additional imports, everything is already there


import numpy as np
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Setting random seed
np.random.seed(0)

# Generating features matrix and target vector
# Number of samples = 1000, Number of features = 3
features, target = make_regression(n_samples = 10000,
                                   n_features = 3,
                                   n_informative = 3,
                                   n_targets = 1,
                                   noise = 0.0,
                                   random_state = 0)

#Divding data into training and test sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.33, random_state=0)


#TO DO ---- 10 POINTS ---------- Create your ANN, Compile it, and Train it ---------------------
""" First Layer (Dense, neurons = 32, Activation = relu)
    Second Layer (Dense, neurons = 32, Activation = relu)
    Third Layer (Dense, neurons = DECIDE YOURSELF, Activation = None)
    Loss (MSE), Optimizer (Adam), Metrics (MSE)
    Epochs (30), Batch Size (100)
"""


# TO DO ---- 1POINT ---- Start neural network
model = models.Sequential()

# TO DO ---- 2 POINTS ---- Add First Layer
# model.add(layers.nn.relu) # add code
model.add(layers.Dense(32, activation='relu'))  # Where should I take relu from, if we used tf.nn.relu???

# TO DO ---- 2 POINTS ---- Add Second Layer

# model.add(layers.Dense(32, activation=activations.relu))
model.add(layers.Dense(32, activation='relu'))


# TO DO ---- 2 POINTS ---- Add Third Layer, You must decide the number of neurons yourself for this layer
model.add(layers.Dense(1, activation=None))


# TO DO ---- 2 POINTS ---- Compile neural network
model.compile(optimizer='adam', loss='MSE', metrics=['MSE'])

# TO DO ---- 2 POINTS ---- Train neural network
model.fit(features_train, target_train, epochs=30, batch_size=100)



#TO DO ---- 5 POINTS ---------- Plot Training and Test Loss ---------------------

# d = model.evaluate(features_test, target_test)
# c = model.evaluate(features_test, target_test)

# plt.plot(d)
# plt.plot(c)
# plt.show()