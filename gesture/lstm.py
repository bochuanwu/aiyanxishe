# encoding: utf-8
"""
@author:  bochuanwu
@contact: wubochuan@xdf.cn
"""
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.datasets import fetch_mldata
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# define constant
num_classes = 4
batch_size = 128
max_epochs = 500
weigth_decay = 0

seed = 1227
files = glob.glob('./train_g/*')

dataframe = []
for index, file in enumerate(files):
    df = pd.read_csv(file)
    dataframe.append(df)

train = pd.concat(dataframe)
train = train.drop('Unnamed: 0', axis=1)
Y = train["64"]
X = train.drop(labels = ["64"],axis = 1)
X = X.values
Y = Y.values

test = pd.read_csv('./test/test.csv', header=None)
results = np.zeros((len(test), num_classes), dtype='float')

import warnings
warnings.filterwarnings('ignore')


def buildlstm():

    import numpy as np

    data_dim = 8
    timesteps = 9
    num_classes = 4

    # expected input data shape: (batch_size, timesteps, data_dim) #32
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,   input_shape=(timesteps, data_dim),dropout_W = 0.1,dropout_U = 0.1))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True,dropout_W = 0.1,dropout_U = 0.1))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    return model


from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

def schedule(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.8
    return lr


def callback_build(weights_file):
    model_ckpt = ModelCheckpoint(os.path.join('./',weights_file), monitor="val_acc", save_best_only=True,
                                 save_weights_only=True, verbose=1,period=1)

    lr_scheduler = LearningRateScheduler(schedule, verbose=0)
    lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=0,
                                     factor=0.8, min_lr=1e-8)
    callback_list = [model_ckpt]
    return callback_list


def smooth_labels(y, smooth_factor=0.05):

    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


models = []
histories = []
scores = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, valid in kfold.split(X, Y):
    scaler = StandardScaler().fit(X[train])
    X_train = scaler.transform(X[train])
    X_valid = scaler.transform(X[valid])
    X_train = np.array([X_train[:, 0:8], X_train[:, 8:16], X_train[:, 16:24], X_train[:, 24:32], X_train[:, 32:40], X_train[:, 40:48], X_train[:, 48:56], X_train[:, 56:64],X_train[:, 0:64:8]]).swapaxes(0, 1)
    X_valid = np.array([X_valid[:, 0:8], X_valid[:, 8:16], X_valid[:, 16:24], X_valid[:, 24:32], X_valid[:, 32:40], X_valid[:, 40:48], X_valid[:, 48:56], X_valid[:, 56:64],X_valid[:, 0:64:8]]).swapaxes(0, 1)


    Y_train = to_categorical(Y[train], num_classes)
    Y_train = smooth_labels(Y_train, smooth_factor=0.05)
    Y_valid = to_categorical(Y[valid], num_classes)

    weights_file = ("best_wegihts_model%d")%len(models) + ".h5"
    callbacks = callback_build(weights_file)
    model = buildlstm()
    history = model.fit(X_train, Y_train,
                        batch_size = batch_size,
                        epochs = max_epochs,
                        callbacks = callbacks,
                        validation_data = [X_valid, Y_valid],
                        shuffle = True)
    # reload best weights of model
    #scaler = StandardScaler().fit(X[train])
    #model = model_build()
    weights_file = ("best_wegihts_model%d") % len(models) + ".h5"
    model.load_weights(os.path.join('./',weights_file), by_name=True)
    print("Model loaded.")
    # evaluate
    #score = model.evaluate(X_valid, Y_valid)
    #scores.append(score)

    X_test = scaler.transform(test)
    X_test = np.array([X_test[:, 0:8], X_test[:, 8:16], X_test[:, 16:24], X_test[:, 24:32], X_test[:, 32:40], X_test[:, 40:48],X_test[:, 48:56], X_test[:, 56:64],X_test[:, 0:64:8]]).swapaxes(0, 1)
    results += model.predict(X_test)
result = np.argmax(results, axis=1)
print(result)
results = pd.Series(result)
submission = pd.concat([pd.Series(range(1, 3504)), results], axis=1)
submission.to_csv('./submission.csv', index=False)