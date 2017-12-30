import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
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
from numba import jit
# from keras.optimizers import RMSprop
import operator
from sklearn.metrics import roc_auc_score
import pickle
from hyperopt.pyll.base import scope 
import tensorflow as tf
import os
import socket
import gc
from scipy.stats import hmean
import tensorflow as tf

host = socket.gethostname()

filename = "keras_params_new.txt".format(host)

iter = 1

# Hyperparameters tuning

#trials = MongoTrials('mongo://34.200.213.137:27017/hyperopt/jobs', exp_key='ensemble')

train_data = np.load('train_data_dae.npy')

test_data = np.load('test_data_dae.npy')

train_target = np.load('train_target.npy')

print(train_data.shape)
print(test_data.shape)

df1 = pd.read_csv('test.csv')

ntrain = train_data.shape[0]
ntest  = test_data.shape[0]

def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_tf_metric(y_true, y_pred):
    return 2.0 * tf.metrics.auc(y_true, y_pred)[0] - 1.0

class roc_callback(keras.callbacks.Callback):

    def __init__(self,training_data,validation_data):
        
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
            
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)      
        
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)      
        
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

class RocAucMetricCallback(keras.callbacks.Callback):

    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.max_delta = min_delta
        self.value = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if(self.include_on_batch):
            logs['roc_auc_val']=float('-inf')
            if(self.validation_data):
                logs['roc_auc_val']= 2.0 * roc_auc_score(self.validation_data[1], self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size)) - 1.0

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val']=float('-inf')
        if(self.validation_data):
            logs['roc_auc_val']= 2.0 * roc_auc_score(self.validation_data[1], self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size)) - 1.0


#, kernel_regularizer=regularizers.l2(0.05)

model = Sequential()
model.add(Dense(units=400, input_dim = train_data.shape[1], kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())


model.add(Dense(units=100, kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(units=100, kernel_initializer=glorot_normal()))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
             
model.add(Dense(1)) 
model.add(Activation('sigmoid'))

opt = keras.optimizers.RMSprop(lr=0.05, rho=0.9, epsilon=1e-08, decay=0.004)

model.compile(loss='binary_crossentropy', optimizer=opt)

model.save("model_clean.h5", overwrite=True)

result = np.zeros(ntrain)

k = 5

#skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=7123)
skf = KFold(n_splits=k, shuffle=True, random_state=34123)

score_fold = np.zeros(k)

i = 0

test_all = np.zeros((ntest,k))

for train_index, test_index in skf.split(train_data, train_target):

    x1 = train_data[train_index]
    x2 = train_data[test_index]
    y1 = train_target[train_index]
    y2 = train_target[test_index]

    print('Round {0} of {1}'.format(i + 1, k))

    collected = gc.collect()
    print ("Function Garbage collector: collected {} objects.".format(collected))

    model = load_model("model_clean.h5")

    opt = keras.optimizers.RMSprop(lr=0.05, rho=0.9, epsilon=1e-08, decay= 3.0 / 150.0 )

    model.compile(loss='binary_crossentropy', optimizer=opt)

    chck = ModelCheckpoint('keras_final_dae.h5', monitor='roc_auc_val', save_best_only=True, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)

    cb = [  RocAucMetricCallback(),  
            reduce_lr,
            EarlyStopping(monitor='val_loss',patience=20, verbose=2, min_delta=0),
            chck
         ]

    model.fit(x1,y1, batch_size=1280, validation_data=(x2, y2), verbose=1, epochs=150, callbacks=cb)
                     
    model = load_model('keras_final_dae.h5')

    x2_pred = model.predict(x2)[:,0]

    test_all[:,i] = model.predict(test_data)[:,0]

    # test_all[:,i] = model.predict(train_data)[:,0]

    score = gini(y2, x2_pred)

    score_fold[i] = score

    print("\t\tScore {0}\n\n".format(score))
    
    i = i + 1

score_final = np.mean(score_fold)

print("\nScore Final {0}".format(score_final))

print("\nScore Result {0}".format(score_all))


filename_save = 'keras_final_dae.csv'

print("Saving {}".format(filename_save))

df1['target'] = np.mean(test_all,axis=1)
df1[['id','target']].to_csv(filename_save, index=False, float_format='%.5f')









