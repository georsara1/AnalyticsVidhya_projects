
#Import modules
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.cluster import KMeans

#Import data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

#---------------------------Pre-processing-------------------------
print('Pre-processing full data set...')
# Concatenate data sets
test_df['stroke'] = 0
train_df['is_train'] = 1
test_df['is_train'] = 0

df_all = pd.concat([train_df, test_df], axis = 0)
df_all = df_all.reset_index(drop = True)

#In null of smoking status if work_type is children replace with never smoked
df_all.smoking_status[df_all.work_type == 'children'] = 'never smoked'

#Create new variable from age binnings
df_all['age_bins'] = pd.cut(df_all.age, 4, precision = 3)

#Create new variable from bmi binnings
df_all['bmi_bins'] = pd.cut(df_all.bmi, 7, precision = 3)


#Set correct data types
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_bins']
for col in categorical_columns:
    df_all[col] = df_all[col].astype('category')

#Dont include smoking status in categorical transformations
to_make_cat_codes = ['gender', 'ever_married', 'work_type', 'Residence_type', 'age_bins']
for col in to_make_cat_codes:
    df_all[col] = df_all[col].cat.codes


#Drop id
df_all = df_all.drop(['id'], axis = 1)

#Split in train and test set
train = df_all[df_all.is_train == 1]
test = df_all[df_all.is_train == 0]

train = train.drop(['is_train'], axis = 1)
test = test.drop(['is_train', 'stroke'], axis = 1)

#-------------------------------------------------------------------------------------------------
train_y = train.stroke
train_x = train.drop(['stroke'], axis = 1)

test_x = test

#------------------------Build LightGBM Model no1-----------------------
train_data=lgb.Dataset(train_x,label=train_y)

#Select Hyper-Parameters
params1 = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 24,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 0.7,
          'subsample_freq': 1,
          'colsample_bytree': 0.64,
          'reg_alpha': 1.2,
          'reg_lambda': 1.4,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 96,
          'num_class' : 1,
          'metric' : 'binary_error'
          }

#Train model on selected parameters and number of iterations
lgbm1 = lgb.train(params1,
                 train_data,
                 220,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm_prob1 = lgbm1.predict(test_x)

#------------------------Build LightGBM Model no2-----------------------

#Select Hyper-Parameters
params2 = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 31,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 0.7,
          'subsample_freq': 1,
          'colsample_bytree': 0.64,
          'reg_alpha': 1.5,
          'reg_lambda': 1.8,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          #'scale_pos_weight': 6.7,
          'is_unbalance': True,
          'num_class' : 1,
          'metric' : 'auc'
          }

#Train model on selected parameters and number of iterations
lgbm2 = lgb.train(params2,
                 train_data,
                 140,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm_prob2 = lgbm2.predict(test_x)

#lgb.plot_importance(lgbm1, max_num_features=21, importance_type='split')
#lgb.plot_importance(lgbm2, max_num_features=21, importance_type='split')
#--------------------------Print accuracy measures and variable importances----------------------
predictions_ens = 0.6 * predictions_lgbm_prob1 + 0.85 * predictions_lgbm_prob2

predictions_ens = predictions_ens / np.max(predictions_ens)

submission_df.stroke = predictions_ens
submission_df.to_csv('lgb_submission.csv', index = False)