__author__ = 'Evgenii Vasilev'

import pandas as pd
import re
import numpy as np
import pickle
from pathlib import Path
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.initializers import RandomNormal
from keras.layers.core import Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop, SGD
import collections



class Text2Gender:

    def __init__(self):
        self.tfidf = pickle.load(open('./preprocessing/tfidf_converter/tfidf_unigram.p','rb'))
        self.logreg = pickle.load(open('./models/logreg/logreg_30m_unigram.p','rb'))


    def to_sparse(self,data):
        """Converts iterable of text to a sparse tf-idf representation

            Parameters
            ----------
            data : iterable
                List (or other iterable) of texts that need to be converted.

            Returns
            -------
            sparse matrix
                Returns a sparse matrix with continues values.
            """
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = [re.sub(r"(?:\@|https?\://)\S+", "", str(x)) for x in data]
        return self.tfidf.transform(data)

    def predict_logreg(self, data):
        """Predicts author gender for provided texts, where 1 is female author and 0 is male

            Parameters
            ----------
            data : iterable
                List (or other iterable) of texts that need to be converted.

            Returns
            -------
            List
                List of probabilities
            """
        if not isinstance(data, collections.Iterable):
            raise Exception("The data should consist of sequence of texts.")

        data = self.to_sparse(data)
        res = self.logreg.predict_proba(data)[:,1]
        rebalanced_coefficient = 0.5/0.52  # The coefficient to re-balance predictions to 0.5 cutoff
        res = res * rebalanced_coefficient
        res = np.clip(res, 0, 1)
        return res

    def predict_xgb(self, data):
        """Predicts gender for provided texts

            Parameters
            ----------
            data : iterable
                List (or other iterable) of texts that need to be converted.

            Returns
            -------
            List
                List of probabilities
            """
        import xgboost as xgb
        
        self.xgbst = xgb.Booster()
        self.xgbst.load_model('./models/xgboost/xgb_25m_unigram.model')
        if not isinstance(data, collections.Iterable):
            raise Exception("The data should consist of sequence of texts.")

        data = self.to_sparse(data)
        res = self.xgbst.predict(xgb.DMatrix(data))
        rebalanced_coefficient = 0.5 / 0.44  # The coefficient to re-balance predictions to 0.5 cutoff
        res = res * rebalanced_coefficient
        res = np.clip(res, 0, 1)
        return res

    def predict_lstm(self,data):
        """Predicts gender for provided texts

               Parameters
               ----------
               data : iterable
                   List (or other iterable) of texts that need to be converted.

               Returns
               -------
               List
                   List of probabilities
        """

        def txt2seq(txt):
            # Sequence length is limited to 50 words
            v = np.zeros(50, dtype='int32')
            try:
                tl = re.sub(r"(?:\@|https?\://)\S+", "", txt.lower()).split()
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

        num_lstm = 256
        num_dense = 512
        rate_drop_lstm = 0.36
        rate_drop_dense = 0.25
        MAX_SEQUENCE_LENGTH = 50
        EMBEDDING_DIM = 300
        nb_words = 2196017 + 1

        word_index = pd.read_csv('./models/lstm/word_index_glove.csv')
        word_index = word_index.set_index('word')['id'].to_dict()

        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

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
                      optimizer='sgd', metrics=['accuracy'])
        #print(model.summary())
        if Path('./models/lstm/lstm_full_data.h5').exists():
            model.load_weights('./models/lstm/lstm_full_data.h5')
        else:
            raise("LSTM weights are missing")
        
        print("LSTM model loaded")

        num_texts = len(data)
        data_converted = np.zeros((num_texts, 50))
        for i in range(num_texts):
            data_converted[i] = txt2seq(data[i])

        res = model.predict([data_converted], batch_size=1024, verbose=0)

        rebalanced_coefficient = 0.5 / 0.51  # The coefficient to re-balance predictions to 0.5 cutoff
        res = res * rebalanced_coefficient
        res = np.clip(res, 0, 1)

        return res

    def predict_charcnn(self, data):
        """Predicts gender for provided texts

               Parameters
               ----------
               data : iterable
                   List (or other iterable) of texts that need to be converted.

               Returns
               -------
               List
                   List of probabilities
        """
        def char2seq(txts):
            X = np.zeros((len(txts), MAX_SEQUENCE_LENGTH), dtype=np.int16)
            for i, sentence in enumerate(txts):
                for t, char in enumerate(sentence.lower()):
                    if t < MAX_SEQUENCE_LENGTH:
                        try:
                            X[i, t] = char_indices[char]
                        except KeyError:
                            X[i, t] = len(chars)
            return X

        def convert_text(comment):
            if type(comment) == str:
                comment = np.array([comment])
            try:
                return char2seq(comment)
            except Exception as e:
                #print(e)
                return char2seq(' ')


        MAX_SEQUENCE_LENGTH = 150  # Maximum number of charachters in comments

        chars = list(""" abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        MAX_NB_CHARS = len(chars) + 1

        embedding_matrix = np.zeros((MAX_NB_CHARS, MAX_NB_CHARS))
        for i in range(1, MAX_NB_CHARS):  # Leave the space as zeros
            embedding_matrix[i][i] = 1

        k_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)

        model = Sequential()
        model.add(Embedding(MAX_NB_CHARS,
                            MAX_NB_CHARS,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))

        # Layer 1
        model.add(
            Convolution1D(input_shape=(MAX_SEQUENCE_LENGTH, MAX_NB_CHARS), filters=256, kernel_size=7, padding='valid',
                          activation='relu', kernel_initializer=k_init))
        model.add(MaxPooling1D(pool_size=3, strides=3))
        # Layer 2
        model.add(
            Convolution1D(filters=256, kernel_size=7, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(MaxPooling1D(pool_size=3, strides=3))
        # Layer 3, 4, 5
        model.add(
            Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(
            Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(
            Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
        # Layer 6
        model.add(
            Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
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
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # print(model.summary())
        
        # Loading pre-trained weights
        if Path('./models/charcnn/cnn_char_from_paper_full_data_184m.h5').exists():
            model.load_weights('./models/charcnn/cnn_char_from_paper_full_data_184m.h5')
        else:
            raise("CharCNN weights are missing")


        if isinstance(data, pd.DataFrame):
            data = data.values
        data = [re.sub(r"(?:\@|https?\://)\S+", "", str(x)) for x in data]

        converted_data = convert_text(data)

        res = model.predict(converted_data, verbose = 0)
        rebalanced_coefficient = 0.5 / 0.475  # The coefficient to re-balance predictions to 0.5 cutoff
        res = res * rebalanced_coefficient
        res = np.clip(res, 0, 1)
        return res












