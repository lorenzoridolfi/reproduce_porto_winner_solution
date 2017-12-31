import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from multiprocessing import *
from sklearn.model_selection import StratifiedKFold, KFold
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn     import metrics
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
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.mongoexp import MongoTrials
from scipy.stats import hmean
import tensorflow as tf
from scipy.special import erfinv

host = socket.gethostname()

filename = "keras_params_dae.txt"

iter = 1

# Hyperparameters tuning

#trials = MongoTrials('mongo://34.200.213.137:27017/hyperopt/jobs', exp_key='ensemble')

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x



train_data = np.load('train_data_dae.npy')

test_data = np.load('test_data_dae.npy')

train_target = np.load('train_target.npy')


ntrain = train_data.shape[0]

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


def score(params):
    print("Training with params: ")
    print(params)

    k = 5

    i = 0

    model = Sequential()
    model.add(Dense(units=4500, input_dim = train_data.shape[1], kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(params['l2'])))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))
    model.add(BatchNormalization())

    
    model.add(Dense(units=1000, kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(params['l2'])))
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout2']))
    model.add(BatchNormalization())

    model.add(Dense(units=1000,kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(params['l2']))) 
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout2']))
    model.add(BatchNormalization())
                 
    model.add(Dense(1)) 
    model.add(Activation('sigmoid'))

    model.save("model_clean.h5", overwrite=True)

    result = np.zeros(ntrain)

    #skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=7123)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=34123)

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

        opt = keras.optimizers.Adam(lr=params['lr'])

        model.compile(loss='binary_crossentropy', optimizer=opt)

        chck = ModelCheckpoint('keras_hyper.h5', monitor='roc_auc_val', save_best_only=True, mode='max')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)

        cb = [  RocAucMetricCallback(),  
#                reduce_lr,
                EarlyStopping(monitor='roc_auc_val',patience=10, verbose=2, min_delta=0, mode='max'),
                chck
             ]

        model.fit(x1,y1, batch_size=params['batch_size'], validation_data=(x2, y2), verbose=1, epochs=100, callbacks=cb)
                         
        model = load_model('keras_hyper.h5')

        x2_pred = model.predict(x2)[:,0]

        # test_all[:,i] = model.predict(train_data)[:,0]

        score = gini(y2, x2_pred)

        result[test_index] = x2_pred

        score_fold[i] = score

        print("\t\tScore {0}\n\n".format(score))
        
        i = i + 1

    score_final = np.mean(score_fold)

    score_all = gini(train_target, result)

    print("\nScore Final {0}".format(score_final))

    print("\nScore Result {0}".format(score_all))

    with open(filename, "a") as myfile:
        myfile.write('{0} {1} {2}\n'.format(score_final, score_all, params))
        myfile.flush()

    # filename_save = 'hyperopt_keras_ensemble_trial_{}'.format(iter)

    # print("Saving {}".format(filename_save))

    # df1['target'] = np.mean(test_all,axis=1)
    # df1[['id','target']].to_csv(filename_save, index=False, float_format='%.5f')

    return {'loss': 1.0 - score_final, 'status': STATUS_OK}


def optimize(random_state=99):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page: 
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

    global iter

    try:
        trials = pickle.load(open("hyperopt_keras_dae.p", "rb"))
        start  = len(trials.trials)
        print("Already have {} trials".format(start))

    except:
        trials = Trials()
        print("Starting new trials file")
        start = 0

    iter = start + 1

    space = {
            #'lr' : hp.loguniform('lr', np.log(0.05), np.log(0.0005)),
                
            'activation' : hp.choice('activation', ['linear', 'relu']),
            'dropout1': hp.uniform('dropout1', 0., 0.3),
            'dropout2': hp.uniform('dropout2', 0., 0.5),

            'nb_epochs'  : 5000,

            'batch_size' : scope.int(hp.loguniform('batch_size', np.log(100.0), np.log(2000.0))),

            'l2' : hp.loguniform('l2', np.log(1.e-9), np.log(1.e-1)),
            'lr' : hp.loguniform('lr', np.log(0.00001), np.log(0.01))

    }

    # Use the fmin function from Hyperopt to find the best hyperparameters

    for i in range(300):

        print('----------------------------------------------------------------------------------------------------')
        if i > 0:
            print(best)

        print('----------------------------------------------------------------------------------------------------')

        collected = gc.collect()
        print ("Optimize Garbage collector: collected {} objects.".format(collected))

        best = fmin(score,
                space=space,
                algo=tpe.suggest,
                max_evals=(start + i + 1),
                trials=trials)

        pickle.dump(trials, open("hyperopt_keras_dae.p", "wb"))




#victor.fonseca@sciensa.com

best = optimize()

print("The best hyperparameters are: ", "\n")
print(best_hyperparams)

score(best_hyperparams,save=True)




