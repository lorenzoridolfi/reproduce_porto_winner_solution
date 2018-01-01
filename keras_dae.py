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
import tensorflow as tf

filename = "keras_dae.txt"

print('Reading data')

train_data_orig = np.load('train_data.npy')

test_data_orig = np.load('test_data.npy')

train_target = np.load('train_target.npy')

print('Original train data with {} samples'.format(train_data_orig.shape[0]))
print('Original test data with {} samples'.format(test_data_orig.shape[0]))

all_data = np.vstack((train_data_orig, test_data_orig))

print('Adding noise')

train_data_noise = np.load('train_data_noise.npy')

test_data_noise = np.load('test_data_noise.npy')

all_data_noise = np.vstack((train_data_noise, test_data_noise))

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

opt = keras.optimizers.Adam(lr=0.001)

model.compile(loss='mse', optimizer=opt)

print('Training neural net')

chck = ModelCheckpoint('keras_dae.h5', monitor='loss', save_best_only=True)

cb = [ EarlyStopping(monitor='loss', patience=100, verbose=2, min_delta=0), chck ]
            
model.fit(all_data_noise, all_data, batch_size=128, verbose=1, epochs=epochs, callbacks=cb)

model = load_model('keras_dae.h5')

print('Applying neural net')

train_data_transform = model.predict(train_data_orig)
test_data_transform = model.predict(test_data_orig)

print('Saving results')

np.save('train_data_dae.npy', train_data_transform)
np.save('test_data_dae.npy', test_data_transform)












