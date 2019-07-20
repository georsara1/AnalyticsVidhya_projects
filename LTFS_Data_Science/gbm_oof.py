import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import time
from datetime import timedelta, date
from tqdm import tqdm

train = pd.read_csv('train.csv')

oof_lgb4 = np.load('oof_lgb4.npy')
oof_lgb4_3 = np.load('oof_lgb4_3.npy')
oof_lgb5 = np.load('oof_lgb5.npy')
oof_lgb7 = np.load('oof_lgb7.npy')
oof_lgb8 = np.load('oof_lgb8.npy')
oof_lgb8_4 = np.load('oof_lgb8_4.npy')
oof_lgb9 = np.load('oof_lgb9.npy')
oof_lgb11 = np.load('oof_lgb11.npy')

train_df = pd.DataFrame({'oof_lgb4': oof_lgb4,
                         'oof_lgb4_3': oof_lgb4_3,
                         'oof_lgb5': oof_lgb5,
                         'oof_lgb7': oof_lgb7,
                         'oof_lgb8': oof_lgb8,
                         'oof_lgb8_4': oof_lgb8_4,
                         'oof_lgb9': oof_lgb9,
                         'oof_lgb11': oof_lgb11,

                         })

test4 = pd.read_csv('submission_lgb4.csv')
test4_3 = pd.read_csv('submission_lgb4_3.csv')
test5 = pd.read_csv('submission_lgb5.csv')
test7 = pd.read_csv('submission_lgb7.csv')
test8 = pd.read_csv('submission_lgb8.csv')
test8_4 = pd.read_csv('submission_lgb8_4.csv')
test9 = pd.read_csv('submission_lgb9.csv')
test11 = pd.read_csv('submission_lgb11.csv')

test_df = pd.DataFrame({'oof_lgb4':test4['loan_default'].values,
                        'oof_lgb4_3':test4_3['loan_default'].values,
                        'oof_lgb5':test5['loan_default'].values,
                        'oof_lgb7':test7['loan_default'].values,
                        'oof_lgb8':test8['loan_default'].values,
                        'oof_lgb8_4':test8_4['loan_default'].values,
                        'oof_lgb9':test9['loan_default'].values,
                        'oof_lgb11':test11['loan_default'].values,})

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb

train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df, train['loan_default'], test_size=0.15)

#Random Forests
print('Random forests...')
clf_rf = RandomForestClassifier(max_depth=7, n_estimators=100)
clf_rf.fit(train_early_x,train_early_y)
pred_rf = clf_rf.predict_proba(valid_early_x)[:,1]

auc_rf = roc_auc_score(valid_early_y,pred_rf)
print('AUC RF: {}'.format(auc_rf))


#Ridge regression
print('Ridge...')
clf_ridge = Ridge()
clf_ridge.fit(train_early_x,train_early_y)
pred_ridge = clf_ridge.predict(valid_early_x)

auc_ridge = roc_auc_score(valid_early_y,pred_ridge)
print('AUC Ridge: {}'.format(auc_ridge))


# #bayes classifier
# clf = GaussianNB()
# clf.fit(train_early_x,train_early_x)
# pred_nb = clf.predict(valid_early_x)
#
# auc_nb = roc_auc_score(valid_early_y,pred_nb)

#LightGBM
print('LightGBM...')
train_data=lgb.Dataset(train_early_x,label=train_early_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          #'max_depth' : 10,
          'objective': 'regression',
          #'nthread': 5,
          'num_leaves': 13,
          'learning_rate': 0.08,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'feature_fraction': 0.4,
          #'subsample': 1,
          #'subsample_freq': 1,
          #'colsample_bytree': 0.8,
          'reg_alpha': 7,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          #'min_child_weight': 1,
          #'min_child_samples': 5,
          'scale_pos_weight': 2.5,
          #'num_class' : 1,
          #'is_unbalance': True,
          'metric' : 'mse',
          #'device' : 'gpu',
          #'gpu_platform_id' : 0,
          #'gpu_device_id' : 0
          }

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 150,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm_prob = lgbm.predict(valid_early_x)

auc_lgb = roc_auc_score(valid_early_y,predictions_lgbm_prob)
print('AUC LGBM: {}'.format(auc_lgb))


#Ensemble and predict on test set
print('Predict on test set...')
test_pred_rf = clf_rf.predict_proba(test_df)[:,1]
test_pred_ridge = clf_ridge.predict(test_df)
test_pred_gbm = lgbm.predict(test_df)

submission = pd.read_csv('sample_submission.csv')
submission['loan_default'] = (test_pred_ridge+test_pred_gbm+test_pred_rf)/3
submission.to_csv('submission_oof_ensemble_1.csv', index = False)


#plots
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(test_pred_ridge, label = 'ridge')
#sns.kdeplot(test_pred_rf, label = 'rf')
sns.kdeplot(test_pred_gbm, label = 'gbm')
#sns.kdeplot(submission['loan_default'].values, label = 'ensemble')