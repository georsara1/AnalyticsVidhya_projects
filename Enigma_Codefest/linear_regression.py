import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score
from sklearn import linear_model
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

from math import sqrt, floor

#Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission_lr = pd.read_csv('sample_submission.csv')

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
#df_all['views_per_rep'] = df_all['Views'] / df_all['Reputation']

del df_all['Username']

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

#----------------Build Linear Regression model------------
regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)

# Train the model using the training sets
regr.fit(train_early_stop_x, train_early_stop_y)

# Make predictions using the testing set
predictions = regr.predict(test)
predictions[predictions<0] = 0

submission_lr = pd.DataFrame({'ID': df_test['ID'], 'Upvotes': predictions})
submission_lr = submission_lr.sort_values(by = 'ID')

submission_lr['Upvotes'] = predictions

df_submission.to_csv('LightGBM_reg_1.csv', index = False)