# Importing packages
import codecs
import csv
import os
import re
from collections import defaultdict
from string import punctuation

import h5py as h5
import numpy as np
import pandas as pd
import sklearn
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.core import Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

num_lstm = 256
num_dense = 512
rate_drop_lstm = 0.36  # 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.25  # 0.15 + np.random.rand() * 0.25

STAMP = "lstm_full_data"

BASE_DIR = './full_data/'
GLOVE_DIR = './full_data/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
VAL_DATA_FILE = BASE_DIR + 'validation_comments.csv'
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

print('Indexing word vectors.')


def load_glove():
    embeddings_index = {}
    f = codecs.open(os.path.join(
        GLOVE_DIR, 'glove.840B.300d.txt'), encoding='utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

embeddings_index = load_glove()

print("GLOVE loaded")

word_index = pd.read_csv('word_index_glove.csv')
word_index = word_index.set_index('word')['id'].to_dict()
nb_words = 2196017 + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
count = 0
for k, v in word_index.items():
    try:
        embedding_matrix[v] = embeddings_index[k]
    except:
        print(k)

f = h5.File('prepared_batches_lstm.hdf5', 'r')
d = f['data']

bind = np.arange(0, d.shape[0]-550, 512)  # Generating indexes for batch parts
np.random.shuffle(bind)
thr = len(bind)-2  # number of half batches
ind = 0


def generate_arrays_from_file():
    global ind
    global bind
    while 1:
        # Combining indexes of two parts of the batch
        cin = bind[ind], bind[ind+1]
        xy = np.concatenate(
            (d[cin[0]:cin[0]+512], d[cin[1]:cin[1]+512]), axis=0)
        x = xy[:, :50]
        y = xy[:, 50]
        ind += 2
        if ind > thr:
            ind = 0
            np.random.shuffle(bind)
        yield (x, y)


def txt2seq(txt):
    # Sequence lenght is limited to 50 words, last integer is for label
    v = np.zeros(50, dtype='int32')
    try:
        tl = txt.lower().split()
        if len(tl) > 50:
            tl = tl[:50]
        ci = 0  # Current word
        for w in tl:
            try:
                v[ci] = word_index[w]
                ci += 1
            except KeyError:
                pass
    except:
        print(txt)
    return v


print("Preparing validation set")
valid = pd.read_csv('./full_data/full_validation.csv')
valid = valid[~pd.isnull(valid.body.values)]
valid = valid[~valid.gender.isnull()]

valn = np.zeros((valid.shape[0], 50))

for i in range(valid.shape[0]):  # Converting validation set to embeddings
    valn[i] = txt2seq(valid.iloc[i, 1])
y_val = valid.gender.values
del valid

print("Data prepared")


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
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm,
                      recurrent_dropout=rate_drop_lstm)

    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_text = embedding_layer(text_input)

    lstm_1 = lstm_layer(embedded_text)

    l_dense = Dense(num_dense, activation='relu')(lstm_1)
    bn4 = BatchNormalization()(l_dense)
    dr4 = Dropout(rate_drop_dense)(bn4)
    preds = Dense(1, activation='sigmoid')(dr4)

    model = Model(inputs=[text_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd', metrics=[f1, 'accuracy'])
    print(model.summary())
    return model

model = load_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=2*10)
bst_model_path = './models/' + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(
    bst_model_path, save_best_only=True, save_weights_only=True)


class_weight = {0: 1.0, 1: 3.697190893835291}
batch_size = 1024
total_comments = d.shape[0]
steps = total_comments//batch_size//10
model.fit_generator(generate_arrays_from_file(),
                    steps_per_epoch=steps, epochs=2000, verbose=2,
                    callbacks=[early_stopping, model_checkpoint],
                    validation_data=([valn], y_val),
                    class_weight=class_weight, shuffle=True)

model.load_weights(bst_model_path)

print("Preparing test set")
test = pd.read_csv('./full_data/full_test.csv')
test = test[~pd.isnull(test.body.values)]
test = test[~test.gender.isnull()]
tstn = np.zeros((test.shape[0], 50))
for i in range(test.shape[0]):
    tstn[i] = txt2seq(test.body.values[i])
y_test = test.gender.values


print("Test F1 score: ", model.evaluate([tstn], y_test, verbose=0))
preds = model.predict([tstn], batch_size=1024, verbose=0)
pd.DataFrame(preds, columns=['proba']).to_csv(
    './models/'+STAMP+'_pred.csv', index=False)
test['pred'] = preds
tstgr = test.groupby('author')
test_mean = tstgr.pred.mean().to_frame()
test_mean = test_mean.reset_index()
labeled_users = pd.read_csv('all_labeled_users2.csv')
lu = labeled_users.set_index('author')['gender'].to_dict()
test_mean['gender'] = test_mean.author.apply(lambda x: lu[x])
test_mean['predicted_gend'] = (test_mean['pred'] > 0.5).apply(lambda x: int(x))
test_mean['gender'] = (test_mean['gender'] == 'f').apply(lambda x: int(x))

ct = 0.4
print("Predictions for different thresholds for " + STAMP)
for i in range(20):
    test_mean['predicted_gend'] = (
        test_mean['pred'] > ct).apply(lambda x: int(x))
    print(ct, 'F1 score: ', f1_score(
        test_mean['gender'], test_mean['predicted_gend']))
    print("###"*5)
    ct = ct + 0.01
