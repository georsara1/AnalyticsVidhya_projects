import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, mean_squared_error
sns.set_style("whitegrid")

df_train = pd.read_csv('all_data.csv')
volume_forecasts = pd.read_csv('volume_forecast_test.csv')
sku_recommendations = pd.read_csv('sku_recommendation_test.csv')

#---------------------------Pre-processing-------------------------
#df.isnull().sum() #No missing values

#df = df.drop(['ID','BILL_AMT4', 'BILL_AMT6'], axis = 1)

#Create two new variables: Year and Month
df_train['YearMonth'] = df_train.YearMonth.astype(str)
df_train['Month'] = df_train.YearMonth.str[4:6]
df_train['Year'] = df_train.YearMonth.str[0:4]
df_train = df_train.drop(['YearMonth'], axis = 1)

#Set correct feature types
df_train['Year'] = df_train.Year.astype(int)
df_train['Month'] = df_train.Month.astype('category')
df_train['Agency'] = df_train.Agency.astype('category')
df_train['SKU'] = df_train.SKU.astype('category')


#Encode categorical variables to ONE-HOT
# print('Converting categorical variables to numeric...')
#
# categorical_columns = ['Agency', 'SKU']
#
# df_train = pd.get_dummies(df_train, columns = categorical_columns,
#                     #drop_first = True #Slightly better performance with n columns in One-Hot encoding
#                     )

#Scaling slightly worsened the results in Gradient Boosting (kept in comments below for reference purposes)
#
# #Scale variables to [0,1] range
# columns_to_scale = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT5'
#     , 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
#
# df[columns_to_scale]=df[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))


#Split in 75% train and 25% test set
train, dev = train_test_split(df_train, test_size = 0.25, random_state= 1984)

train_y = train.Volume
dev_y = dev.Volume

train_x = train.drop(['Volume'], axis = 1)
dev_x = dev.drop(['Volume'], axis = 1)

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data = lgb.Dataset(dev_x, label= dev_y)

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
          'metric' : 'root_mean_squared_error'
          }

# Create parameters to search
gridParams = {
    'learning_rate': [0.05],
    'n_estimators': [8,16],
    'num_leaves': [20, 24, 31],
    'boosting_type' : ['gbdt'],
    'objective' : ['regression'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.64, 0.65],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1, 1.2],
    'reg_lambda' : [ 1.2, 1.4],
    }

# Create classifier to use. Note that parameters have to be input manually, not as a dict!
mdl = lgb.LGBMRegressor(boosting_type= 'gbdt',
          objective = 'regression',
          n_jobs = 5,
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

# To view the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)

# Run the grid
grid.fit(train_x, train_y)

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
                 18000,
                 early_stopping_rounds= 40,
                 valid_sets= [valid_data],
                 verbose_eval= 4
                 )

#Then we must re-train our model on the entire dataset
train_y = df_train.Volume
train_x = df_train.drop(['Volume'], axis = 1)

train_data=lgb.Dataset(train_x,label=train_y)

params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'regression',
          'nthread': 5,
          'num_leaves': 31,
          #'n_estimators': 16,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 0.7,
          'subsample_freq': 1,
          'colsample_bytree': 0.64,
          'reg_alpha': 1.6,
          'reg_lambda': 1.6,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'mean_squared_error'
          }

lgbm = lgb.train(params,
                 train_data,
                 297,
                 #early_stopping_rounds= 40,
                 #valid_sets= [valid_data],
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm = lgbm.predict(df_test)

volume_forecasts.Volume = predictions_lgbm
volume_forecasts.Volume[volume_forecasts.Volume<0] = 0

sku_recommendations.SKU[0] = 'SKU_01'
sku_recommendations.SKU[1] = 'SKU_04'
sku_recommendations.SKU[2] = 'SKU_01'
sku_recommendations.SKU[3] = 'SKU_02'

volume_forecasts.to_csv('volume_forecast.csv', index= False)
sku_recommendations.to_csv('sku_recommendation.csv', index= False)