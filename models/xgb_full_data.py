import gc
import pickle
import re
import string

import h5py as h5
import numpy as np
import pandas as pd
import scipy
import sklearn.metrics
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

BASE_DIR = '../full_data/'
TEST_DATA_FILE = BASE_DIR + 'full_test.csv'
VAL_DATA_FILE = BASE_DIR + 'full_validation.csv'

test = pd.read_csv(TEST_DATA_FILE)
test = test[~pd.isnull(test.body.values)]
test = test[~test.gender.isnull()]
val = pd.read_csv(VAL_DATA_FILE)
print("Val shape", val.shape)
val = val[~pd.isnull(val.body.values)]
val = val[~val.gender.isnull()]
print("Val_shape without nan", val.shape)

f = h5.File('../training_data_full.hdf5', 'r')
d = f['data']

train = scipy.sparse.load_npz('train_half.npz')  # Loading data that has been converted by TFIDFVectorizer
train = train[:30000000] # Limitations of memory
print("Shape of train:", train.shape)

y_train = np.array([int(i) for i in d[:train.shape[0], 4]])
f.close()
y_test = test.gender.values
y_val = np.int64(val.gender.values)

val = scipy.sparse.load_npz('val_full.npz')
del test

gc.collect()
print("Data imported")
# Set parameters for xgboost that were chosen with a GridSearchCV on a sample data
params = {'min_child_weight': 1, 'eval_metric': 'logloss', 'eta': 0.3, 'scale_pos_weight': 2.5178, 'subsample': 1,
          'colsample_bytree': 0.8, 'nthread': 24, 'max_depth': 8, 'objective': 'binary:logistic'}


d_train = xgb.DMatrix(train, label=y_train)
print("Train transformed to DMatrix")
d_valid = xgb.DMatrix(val, label=y_val)
print("Data converted to DMatrix")
watchlist = [(d_train, 'train'), (d_valid, 'valid')]


bst = xgb.train(params, d_train, 2000, watchlist,
                early_stopping_rounds=50, verbose_eval=10)
bst.save_model('xgb_half_data.model')

del d_train
del d_valid
gc.collect()

# Loading test data
test = scipy.sparse.load_npz('test_full.npz')

preds = bst.predict(xgb.DMatrix(test))
y_hat = preds > 0.5
y_hat = y_hat.astype(int)
print("F1 score for 0.5 cut", sklearn.metrics.f1_score(y_test, y_hat))

test = pd.read_csv(TEST_DATA_FILE)
test['pred'] = preds
test[['pred']].to_csv('xgb_preds_full.csv')
tstgr = test.groupby('author')
test_mean = tstgr.pred.mean().to_frame()
test_mean = test_mean.reset_index()
labeled_users = pd.read_csv('../all_labeled_users2.csv')
lu = labeled_users.set_index('author')['gender'].to_dict()
test_mean['gender'] = test_mean.author.apply(lambda x: lu[x])
test_mean['gender'] = (test_mean['gender']=='f').apply(lambda x: int(x)) # Converting gender to int

ct = 0.3
print("Predictions for different thresholds")
for i in range(40):
    test_mean['predicted_gend'] = (
        test_mean['pred'] > ct).apply(lambda x: int(x))
    print(ct, 'F1 score: ', f1_score(
        test_mean['gender'], test_mean['predicted_gend']))
    print("###"*5)
    ct = ct + 0.01