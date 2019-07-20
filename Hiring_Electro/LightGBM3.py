#---------------------------------------Import libraries-------------------------------
print('Importing needed libraries...')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import lightgbm as lgb
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import preprocessing

#--------------------------------------Import data------------------------------------
print('Importing data...')
train_set = pd.read_csv('train.csv', sep = ',')

test_set = pd.read_csv('test.csv', sep = ',')
#submission_df = pd.read_csv('sample_submission.csv', sep = ',')

train_labels = train_set['electricity_consumption']

train_set = train_set.drop(["electricity_consumption"], axis=1)

IDs = test_set['ID']
#-----------------------------Exploratory Data Analysis----------------------------
#Merge into one data set for data wrangling
train_set['is_train'] = 1
test_set['is_train'] = 0

train_test_set = pd.concat([train_set, test_set], axis=0, ignore_index=True)

train_test_set = train_test_set.drop(["ID"], axis=1)

#Create variable Year
train_test_set['Year'] = train_test_set.datetime.str[0:4]

#Create variable Month
train_test_set['Month'] = train_test_set.datetime.str[5:7]

#Create variable Day
train_test_set['Day'] = train_test_set.datetime.str[9:11]

#Create variable Hour
train_test_set['Hour'] = train_test_set.datetime.str[11:13]

#Delete original datetime variable
train_test_set = train_test_set.drop(["datetime"], axis=1)

#Set as categorical the respectful variables
train_test_set['Year'] = train_test_set['Year'].astype('category')
train_test_set['Month'] = train_test_set['Month'].astype('category')
train_test_set['Day'] = train_test_set['Day'].astype('category')
train_test_set['Hour'] = train_test_set['Hour'].astype('category')
train_test_set['var2'] = train_test_set['var2'].astype('category')

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = train_test_set.select_dtypes(include=['number']).copy()
var_non_numeric = train_test_set.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

train_test_set= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Split again in train and test sets
train_set = train_test_set.loc[train_test_set['is_train'] == 1]
test_set = train_test_set.loc[train_test_set['is_train'] == 0]

train_set = train_set.drop(['is_train'], axis = 1)
test_set = test_set.drop(['is_train'], axis = 1)

train_set = preprocessing.scale(train_set)
test_set = preprocessing.scale(test_set)

#Build LightGBM Model
train_data=lgb.Dataset(train_set,label=train_labels)

param = {'num_leaves': 24, 'objective':'regression_l2', 'max_depth':10,
         'learning_rate':.08,  'metric': 'rmse',
         'feature_fraction': 0.8}

cv_mod = lgb.cv(param,
                train_data,
                num_boost_round = 1000,
                #min_data = 1,
                nfold = 5,
                early_stopping_rounds = 20,
                verbose_eval=100,
                stratified = True,
                show_stdv=True,
                )


num_boost_rounds_lgb = len(cv_mod['rmse-mean'])

lgbm = lgb.train(param, train_data, num_boost_rounds_lgb)

ax = lgb.plot_importance(lgbm, max_num_features=21)
plt.show()

predictions = lgbm.predict(test_set)

lgbm_submission = pd.DataFrame({'ID': IDs, 'electricity_consumption': predictions})

lgbm_submission.to_csv('lgbm_submission3.csv', sep = ',',index = False)