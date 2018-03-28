import codecs
import csv
import os
import random
import re
import string
import sys
from collections import Counter
from functools import reduce

import h5py as h5
import numpy as np
import pandas as pd
import sklearn
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

f = h5.File('file_path', 'r')
d = f['data']

num_conv = 256
num_dense = 2048

rate_drop_dense = 0.25  # 0.15 + np.random.rand() * 0.25


STAMP = 'cnn_char_full_data'

BASE_DIR = './full_data/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'full_test.csv'
VAL_DATA_FILE = BASE_DIR + 'full_validation.csv'
MAX_SEQUENCE_LENGTH = 150  # Maximum number of characters in comments


# Producing character mapping
chars = list(
    """ abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Number of allowed chars
MAX_NB_CHARS = len(chars)+1


def char2seq(txts, maxlen):
    X = np.zeros((len(txts), maxlen), dtype=np.int16)
    for i, sentence in enumerate(txts):
        for t, char in enumerate(sentence.lower()):
            if t < maxlen:
                try:
                    X[i, t] = char_indices[char]
                except KeyError:
                    X[i, t] = len(chars)
    return X


val = pd.read_csv(VAL_DATA_FILE)
print("Val shape", val.shape)
val = val[~pd.isnull(val.body.values)]
val = val[~val.gender.isnull()]
print("Val_shape without nan", val.shape)

y_val = np.int64(val.gender.values)

val_texts = [i.lower() for i in val.body.values]

data_val = char2seq(val_texts, MAX_SEQUENCE_LENGTH)

embedding_matrix = np.zeros((MAX_NB_CHARS, MAX_NB_CHARS))
for i in range(1, MAX_NB_CHARS):  # Leave the space as zeros
    embedding_matrix[i][i] = 1

bind = np.arange(0, 184823488, 512)
np.random.shuffle(bind)  # Randomizing the batches

ind = 0

def generate_arrays_from_file():  # Generating batches from a hdf5 file
    global ind
    global bind
    while 1:
        cin = bind[ind], bind[ind+1]
        xy = np.concatenate(
            (d[cin[0]:cin[0]+512], d[cin[1]:cin[1]+512]), axis=0)
        x = xy[:, :150]
        y = xy[:, 150]
        ind += 2
        if ind > 360980:
            ind = 0
            np.random.shuffle(bind)
        yield (x, y)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def load_model():
    """"
    The architecture of the model is inspired by from 'Character-level Convolutional Networks for Text Classification'
     Xiang Zhang, Junbo Zhao, Yann LeCun https://arxiv.org/abs/1509.01626
     """
    k_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    model = Sequential()

    model.add(Embedding(MAX_NB_CHARS,
                        MAX_NB_CHARS,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))

    # Layer 1
    model.add(Convolution1D(input_shape=(MAX_SEQUENCE_LENGTH, MAX_NB_CHARS), filters=256,
                            kernel_size=7, padding='valid', activation='relu', kernel_initializer=k_init))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    # Layer 2
    model.add(Convolution1D(filters=256, kernel_size=7, padding='valid',
                            activation='relu', kernel_initializer=k_init))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    # Layer 3, 4, 5
    model.add(Convolution1D(filters=256, kernel_size=3, padding='valid',
                            activation='relu', kernel_initializer=k_init))
    model.add(Convolution1D(filters=256, kernel_size=3, padding='valid',
                            activation='relu', kernel_initializer=k_init))
    model.add(Convolution1D(filters=256, kernel_size=3, padding='valid',
                            activation='relu', kernel_initializer=k_init))
    # Layer 6
    model.add(Convolution1D(filters=256, kernel_size=3, padding='valid',
                            activation='relu', kernel_initializer=k_init))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    # Layer 7
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer=k_init))
    model.add(Dropout(0.5))
    # Layer 8
    model.add(Dense(1024, activation='relu', kernel_initializer=k_init))
    model.add(Dropout(0.5))
    # Layer 9
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=[f1, 'accuracy'])
    print(model.summary())
    print(STAMP)
    return model

model = load_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = './models/' + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(
    bst_model_path, save_best_only=True, save_weights_only=True)
class_weight = {0: 1.0, 1: 3.697190893835291}
print("#### Starting to train ####")

batch_size = 1024
total_comments = d.shape[0]
steps = total_comments//batch_size
model.fit_generator(generate_arrays_from_file(),
                    steps_per_epoch=steps, epochs=200, verbose=2,
                    callbacks=[early_stopping, model_checkpoint],
                    validation_data=([data_val], y_val),
                    class_weight=class_weight, shuffle=True)

model.load_weights(bst_model_path)

test = pd.read_csv(TEST_DATA_FILE)
y_test = test.gender.values
test_texts = [i.lower() for i in test.body.values]
data_test = char2seq(test_texts, MAX_SEQUENCE_LENGTH)

print("Test F1 score: ", model.evaluate([data_test], y_test, verbose=0))
preds = model.predict([data_test], batch_size=1024, verbose=0)
pd.DataFrame(preds, columns=['proba']).to_csv(
    './full_data/'+STAMP+'_pred.csv', index=False)
test['pred'] = preds
tstgr = test.groupby('author')
test_mean = tstgr.pred.mean().to_frame()
test_mean = test_mean.reset_index()
labeled_users = pd.read_csv('all_labeled_users2.csv')
lu = labeled_users.set_index('author')['gender'].to_dict()
test_mean['gender'] = test_mean.author.apply(lambda x: lu[x])
test_mean['predicted_gend'] = (test_mean['pred'] > 0.5).apply(lambda x: int(x))
test_mean['gender'] = (test_mean['gender'] == 'f').apply(lambda x: int(x))

ct = 0.3
for i in range(80): # Checking F1 with different thresholds
    test_mean['predicted_gend'] = (
        test_mean['pred'] > ct).apply(lambda x: int(x))
    print(ct, 'F1 score: ', f1_score(
        test_mean['gender'], test_mean['predicted_gend']))
    print("###"*5)
    ct = ct + 0.005
