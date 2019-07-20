import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from math import sqrt, floor

#Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sample_submission.csv')

df_test['Upvotes'] = 0

df_train['is_train'] = 1
df_test['is_train'] = 0

#Concatenate into a single data frame
df_all = pd.concat([df_train, df_test], axis = 0)

#Check for null values
df_all.isnull().sum().sum() #OK all imputed


#Feature engineering and transformations
df_all['Username'] = df_all['Username'].astype('object')

for col in df_all.columns:
    if df_all[col].dtype == 'object':
        df_all[col] = df_all[col].astype('category')
        df_all[col] = df_all[col].cat.codes


del df_all['ID']

df_all['answer_per_view'] = df_all['Answers'] / df_all['Views']
df_all['views_per_rep'] = df_all['Views'] / df_all['Reputation']

#del df_all['Username']

df_all['Tag_Frequency'] = (df_all.groupby(['Tag'])['Tag'].transform('count'))/df_all.shape[0]

#Split in train, validation and test sets
train = df_all[df_all['is_train'] == 1]
test = df_all[df_all['is_train'] == 0]

del train['is_train']
del test['is_train']

train_early_stop_x, valid_early_stop_x = train_test_split(train, test_size= 0.15, random_state= 78)

train_early_stop_y = train_early_stop_x['Upvotes']
del train_early_stop_x['Upvotes']

valid_early_stop_y = valid_early_stop_x['Upvotes']
del valid_early_stop_x['Upvotes']

del test['Upvotes']

#Build model
train_data=lgb.Dataset(train_early_stop_x,label=train_early_stop_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'regression',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'eval_metric' : 'rmse'
          }

# Create parameters to search
gridParams = {
    'learning_rate': [0.01, 0.05],
    'n_estimators': [8,16, 24],
    'num_leaves': [16, 24, 32],
    'boosting_type' : ['gbdt'],
    'objective' : ['regression'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.65],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [0.1, 1.2, 4],
    'reg_lambda' : [ 0.2, 1.4, 5],
    'eval_metric' : ['rmse']
    }

# Create classifier to use
mdl = lgb.LGBMRegressor(boosting_type= 'gbdt',
          objective = 'regression',
          n_jobs = 5, # Updated from 'nthread'
          silent = False,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

# View the default model params:
mdl.get_params().keys()

# Create the grid
grid = RandomizedSearchCV(mdl, gridParams, verbose=2, cv=8, n_jobs=-1)

# Run the grid
grid.fit(train_early_stop_x, train_early_stop_y,eval_metric= 'rmse')

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']
# params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

print('Fitting with params: ')
print(params)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 4200,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions = lgbm.predict(valid_early_stop_x)
predictions = [floor(p) if p>=0 else 0 for p in predictions]

valid_rmse = sqrt(mean_squared_error(valid_early_stop_y, predictions))


#---------------Train in all data, predict test set and write to file for submission-------
train_x = train.drop(['Upvotes'], axis = 1)
train_y = train['Upvotes']

train_final = lgb.Dataset(train_x,label=train_y)

#Train with Bayes parameters
lgbm = lgb.train(opt_params,
                 train_data,
                 150,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions = lgbm.predict(test)
predictions = [floor(p) if p>=0 else 0 for p in predictions]

df_submission['Upvotes'] = predictions

df_submission.to_csv('LightGBM_bayes_1.csv', index = False)



