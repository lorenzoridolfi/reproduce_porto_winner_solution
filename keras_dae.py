import numpy as np
import pandas as pd
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras.initializers import glorot_normal, Zeros, Ones
import keras.backend as K
from keras.optimizers import RMSprop
import operator
from sklearn.metrics import roc_auc_score
import pickle
from hyperopt.pyll.base import scope 
import tensorflow as tf



filename = "keras_dae.txt"


def inputSwapNoise(arr, p):
    ### Takes a numpy array and swaps a row of each 
    ### feature with another value from the same column with probability p
    n, m = arr.shape
    idx = range(n)
    swap_n = round(n*p)
    for i in range(m):
        col_vals = np.random.permutation(arr[:, i])
        swap_idx = np.random.choice(idx, size= swap_n)
        arr[swap_idx, i] = np.random.choice(col_vals, size = swap_n)
    return arr

print('Reading data')

train_data_orig = np.load('train_data.npy')

test_data_orig = np.load('test_data.npy')

train_target = np.load('train_target.npy')

ntrain = train_data_orig.shape[0]

all_data = np.vstack((train_data_orig, test_data_orig))

all_data_error = np.zeros((all_data.shape))

print('Adding noise')

all_data_error = inputSwapNoise(all_data, 0.15)

ntrain = train_data_orig.shape[0]
ntest  = test_data_orig.shape[0]

print('Creating neural net')

model = Sequential()
model.add(Dense(units=1500, input_dim = all_data.shape[1], kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(units=1500, kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(units=1500, kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(BatchNormalization())
             
model.add(Dense(all_data.shape[1])) 
model.add(Activation('linear'))

epochs = 1000

opt = keras.optimizers.RMSprop(lr=0.02, rho=0.9, epsilon=1e-08, decay= 4 / epochs)

model.compile(loss='mse', optimizer=opt)

print('Training neural net')

chck = ModelCheckpoint('keras_dae.h5', monitor='loss', save_best_only=True)

cb = [ EarlyStopping(monitor='loss',patience=20, verbose=2, min_delta=0), chck ]
            
model.fit(all_data_error, all_data, batch_size=128, verbose=1, epochs=epochs, callbacks=cb)

model = load_model('keras_dae.h5')

print('Applying neural net')

all_data_transform = model.predict(all_data)

print('Saving results')

train_data_transform = all_data_transform[0:ntrain,:]
test_data_transform = all_data_transform[ntrain:,:]

np.save('train_data_dae.npy', train_data_transform)
np.save('test_data_dae.npy', test_data_transform)












