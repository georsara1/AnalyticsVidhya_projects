import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, mean_squared_error, f1_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from math import sqrt, floor

#Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#df_submission = pd.read_csv('sample_submission.csv')

df_test['is_promoted'] = 0

#Concatenate into a single data frame
df_all = pd.concat([df_train, df_test], axis = 0)

#Check for null values
df_all.isnull().sum().sum() #OK all imputed


#Feature engineering and transformations
#df_all['Username'] = df_all['Username'].astype('object')

for col in df_all.columns:
    if df_all[col].dtype == 'object':
        df_all[col] = df_all[col].astype('category')
        df_all[col] = df_all[col].cat.codes


del df_all['employee_id']
del df_all['gender']
#df_all['perc_years_in_company'] = df_all['length_of_service'] / df_all['age']
#df_all['answer_per_view'] = df_all['Answers'] / df_all['Views']
#df_all['views_per_rep'] = df_all['Views'] / df_all['Reputation']

#del df_all['Username']

#Split in train, validation and test sets
train = df_all.iloc[:df_train.shape[0],:]
test = df_all.iloc[df_train.shape[0]:,:]

train_early_stop_x, valid_early_stop_x = train_test_split(train, test_size= 0.15, random_state= 7)

train_early_stop_y = train_early_stop_x['is_promoted']
del train_early_stop_x['is_promoted']

valid_early_stop_y = valid_early_stop_x['is_promoted']
del valid_early_stop_x['is_promoted']

del test['is_promoted']

#Build model
train_data=lgb.Dataset(train_early_stop_x,label=train_early_stop_y)
valid_data=lgb.Dataset(valid_early_stop_x,label=train_early_stop_y)

def lgbm_evaluate(max_depth, colsample_bytree, num_leaves, reg_alpha, reg_lambda):

    params = {'learning_rate': 0.01,
              'objective': 'binary',
              'metric': 'binary_logloss',
              'max_depth': int(max_depth),
              'num_leaves': int(num_leaves),
              'colsample_bytree': colsample_bytree,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'verbose' : -1
             }

    cv_result = lgb.cv(params, train_data, num_boost_round=100, nfold=5, metrics=['binary_logloss'], verbose_eval=-1)

    return max(cv_result['binary_logloss-mean'])


lgbBO = BayesianOptimization(lgbm_evaluate, {'max_depth': (4, 10),
                                             'colsample_bytree': (0.3, 0.9),
                                             'num_leaves': (18,32),
                                             'reg_alpha' : (0, 5),
                                             'reg_lambda' : (0, 5)
                                             })

lgbBO.maximize(init_points=3, n_iter=5)

opt_params = lgbBO.res['max']['max_params']

print(opt_params)

opt_params['max_depth'] = int(opt_params['max_depth'])
opt_params['num_leaves'] = int(opt_params['num_leaves'])

#Train with Bayes parameters
lgbm = lgb.train(opt_params,
                 train_data,
                 500,
                 #valid_sets=[train_data],
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

lgb.plot_importance(lgbm, max_num_features=21, importance_type='gain')
plt.show()

#Predict on test set
predictions = lgbm.predict(valid_early_stop_x)
predictions = [1 if p>=0.5 else 0 for p in predictions]

valid_f1 = f1_score(valid_early_stop_y, predictions)
print('Validation F1 score:', valid_f1)

#Train in all data, predict test set and write to file for submission
train_x = train.drop(['is_promoted'], axis = 1)
train_y = train['is_promoted']

train_final = lgb.Dataset(train_x,label=train_y)

#Train with Bayes parameters
lgbm = lgb.train(opt_params,
                 train_final,
                 600,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions = lgbm.predict(test)
predictions = [1 if p>=0.5 else 0 for p in predictions]

df_submission = pd.DataFrame()
df_submission['employee_id'] = df_test['employee_id']
df_submission['is_promoted'] = predictions

df_submission.to_csv('LightGBM_bayes_1.csv', index = False)
