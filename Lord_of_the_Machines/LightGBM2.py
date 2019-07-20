
#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from datetime import datetime
sns.set_style("whitegrid")
np.random.seed(697)
import lightgbm as lgb

#Import data
print('Importing data...')
#train_df = pd.read_csv('train.csv')
#test_df = pd.read_csv('test.csv')
#campaign_data_df = pd.read_csv('campaign_data.csv')
submission_df = pd.read_csv('sample_submission.csv')
df_all_final = pd.read_pickle('df_all_final_pickled')

#train_df['is_train'] = 1
#test_df['is_train'] = 0

#train_y = df_all_final[df_all_final.is_train == 1].is_click

# df_all_final = df_all_final.drop(['is_click',
#                                   #'campaign_id', 'user_id', 'send_date', 'time', 'index', 'id'
#                                   #'no_of_sections', 'email_url', 'communication_type',
#                                   ], axis=1)

#Split in train and test set
train_x = df_all_final[df_all_final.is_train == 1]
test_x = df_all_final[df_all_final.is_train == 0]

#train_x2 = train_x.drop_duplicates(subset='total_links')
#test_x2 = test_x.drop_duplicates(subset='total_links')

train_y = train_x.is_click

train_x = train_x.drop(['is_train', 'is_click'], axis=1)
test_x = test_x.drop(['is_train', 'is_click'], axis=1)

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
test_data=lgb.Dataset(test_x)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 48,
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
          #'scale_pos_weight': 18,
          'num_class' : 1,
          'metric' : 'auc',
          'is_unbalance' : 'True'
          }

# Create parameters to search
gridParams = {
    'learning_rate': [0.05],
    'n_estimators': [8, 16],
    'num_leaves': [ 16, 24],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [51],
    'colsample_bytree' : [0.6],
    'subsample' : [0.7],
    'reg_alpha' : [1],
    'reg_lambda' : [ 1, 1.2],
    }

# Create classifier to use. Note that parameters have to be input manually, not as a dict!
mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 3,
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          #scale_pos_weight = params['scale_pos_weight']
         )

# To view the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=4)

# Run the grid
grid.fit(train_x, train_y)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
#params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']
#params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

print('Fitting with params: ')
print(params)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 650,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm_prob = lgbm.predict(test_x)
predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
plt.hist(predictions_lgbm_prob)
predictions_lgbm_01.sum()/len(predictions_lgbm_01)

#--------------------------Print accuracy measures and variable importances----------------------
#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')

submission_df.is_click = predictions_lgbm_prob
submission_df.to_csv('lgb_submission.csv', index = False)