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

print('Reading data')

train_data_orig = np.load('train_data.npy')

test_data_orig = np.load('test_data.npy')

print('Loading neural net')

model = load_model('keras_refine_dae.h5')

print('Applying neural net')

train_data_transform = model.predict(train_data_orig)
test_data_transform = model.predict(test_data_orig)

print('Saving results')

np.save('train_data_dae.npy', train_data_transform)
np.save('test_data_dae.npy', test_data_transform)












