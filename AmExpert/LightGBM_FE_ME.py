#[144]	training's auc: 0.615285	valid_1's auc: 0.610407 PL 0.5701

#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold

sns.set_style("whitegrid")


#Import data
print('import data')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
historical_data_df = pd.read_csv('historical_user_logs.csv')
submission_df = pd.read_csv('sample_submission.csv')


label_y = train_df['is_click']
del train_df['is_click']

#---------------------Preprocess Train set--------------------
print('train preprocess')
#Extract time
#train_df['hour'] = train_df['DateTime'].str.slice(11,13)
#train_df['hour'] = train_df['hour'].astype(int)

#Aggregate historical data features
del historical_data_df['DateTime']
# hist_agg_unique = historical_data_df.groupby('user_id').nunique()[['product', 'action']]
# hist_agg_unique.columns = ['product_nunique', 'action_nunique']
#
hist_agg_count = pd.DataFrame(historical_data_df.groupby('product').count())
#hist_agg_count.columns = ['visit_count']
hist_agg_count['popularity'] = hist_agg_count['action']/hist_agg_count['action'].sum()
hist_agg_count = hist_agg_count.drop(['user_id', 'action'], axis = 1)
#
# train_df = train_df.merge(right=hist_agg_unique.reset_index(), how='left', on='user_id')
# train_df = train_df.merge(right=hist_agg_count.reset_index(), how='left', on='user_id')

prod_interest = pd.crosstab(historical_data_df['product'], historical_data_df['action'], normalize='index')
prod_interest['interest'] = prod_interest['interest']*100
del prod_interest['view']

train_df = train_df.merge(right=prod_interest.reset_index(), how='left', on='product')
train_df = train_df.merge(right=hist_agg_count.reset_index(), how='left', on='product')

#Set as categorical the respectful variables
cat_cols = ['user_id', 'product',
            #'campaign_id',
            'webpage_id', 'product_category_1',
            #'product_category_2',
            #'user_group_id',
            #'gender',
            'age_level',
            'user_depth',
            #'city_development_index',
            # 'var_1',
            #'hour',
            ]

for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    train_df[col] = train_df[col].cat.codes

#Replace -1 with null
train_df[train_df==-1] = np.nan

#Drop unneeded features
train_df = train_df.drop(['session_id',
                          'campaign_id',
                      'DateTime',
                      #'hour',
                      #'gender',
                      #'user_id',
                      #'user_group_id',
                      'product_category_2',
                      'city_development_index',
                      #'age_level',
                      #'webpage_id'
                      #'visit_count'
                      ], axis=1)

#---------------------Preprocess Test set--------------------
print('test preprocess')
#Extract time
# test_df['hour'] = test_df['DateTime'].str.slice(11,13)
# test_df['hour'] = test_df['hour'].astype(int)

#Aggregate historical data features
#aggregations have been calculate in train set

# test_df = test_df.merge(right=hist_agg_unique.reset_index(), how='left', on='user_id')
# test_df = test_df.merge(right=hist_agg_count.reset_index(), how='left', on='user_id')

test_df = test_df.merge(right=prod_interest.reset_index(), how='left', on='product')
test_df = test_df.merge(right=hist_agg_count.reset_index(), how='left', on='product')


for col in cat_cols:
    test_df[col] = test_df[col].astype('category')
    test_df[col] = test_df[col].cat.codes

#Replace -1 with null
test_df[test_df==-1] = np.nan

#Drop unneeded features
test_df = test_df.drop(['session_id',
                        'campaign_id',
                      'DateTime',
                      #'hour',
                      'gender',
                      #'user_id',
                      #'user_group_id',
                      'product_category_2',
                      'city_development_index',
                      #'age_level',
                      #'webpage_id'
                      #'visit_count'
                      ], axis=1)


#Frequency encoding
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

len_train = train_df.shape[0]
df_all = pd.concat([train_df, test_df])

for col in tqdm(cat_cols):
    df_all = frequency_encoding(df_all, col)

train_df = df_all[:len_train]
test_df = df_all[len_train:]




#Split in train and validation set
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df, label_y, test_size=0.2, stratify=label_y)


#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_early_x,label=train_early_y)
valid_data=lgb.Dataset(valid_early_x,label=valid_early_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 8,
          'objective': 'binary',
          'nthread': 4,
          'n_estimators': 144,
          'num_leaves': 21,
          'learning_rate': 0.02,
          #'max_bin': 512,
          #'subsample_for_bin': 200,
          #'subsample': 0.7,
          #'subsample_freq': 5,
          #'colsample_bytree': 0.9,
          'reg_alpha': 0.11,
          'reg_lambda': 0.1,
          'min_child_weight': 0.9,
          #'min_child_samples': 5,
          #'scale_pos_weight': 1,
          #'num_class' : 2,
          'metric' : 'auc',
          'is_unbalance' : 'True'
          }

# # Create parameters to search
# gridParams = {
#     'max_depth': [4,6,8],
#     'learning_rate': [0.05, 0.01],
#     #'n_estimators': [14, 40, 80],
#     'num_leaves': [ 12, 16, 24],
#     'boosting_type' : ['gbdt'],
#     'objective' : ['binary'],
#     'random_state' : [501],
#     'colsample_bytree' : [0.3, 0.9],
#     'subsample' : [0.95,0.99],
#     'subsample_freq': [1, 3, 5],
#     #'min_split_gain': [0.2, 0.5],
#     'min_child_weight': [0.8, 0.9],
#     'reg_alpha' : [0.1, 1.2],
#     'reg_lambda' : [0.1, 1.2],
#     }
#
# # Create classifier to use. Note that parameters have to be input manually, not as a dict!
# mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
#           objective = 'binary',
#           n_jobs = 3,
#           silent = False,
#           max_depth = params['max_depth'],
#           #n_estimators = params['n_estimators'],
#           num_leaves = params['num_leaves'],
#           learning_rate = params['learning_rate'],
#           #subsample = params['subsample'],
#           subsample_freq = params['subsample_freq'],
#           #colsample_bytree = params['colsample_bytree'],
#           #min_split_gain = params['min_split_gain'],
#           min_child_weight = params['min_child_weight'],
#           reg_alpha = params['reg_alpha'],
#           reg_lambda = params['reg_lambda'],
#           #min_child_samples = params['min_child_samples'],
#           #scale_pos_weight = params['scale_pos_weight']
#          )
#
# # To view the default model params:
# mdl.get_params().keys()
#
# # Create the grid
# grid = RandomizedSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=32)
#
# # Run the grid
# grid.fit(train_early_x, train_early_y)
#
# # Print the best parameters found
# print(grid.best_params_)
# print(grid.best_score_)
#
# # Using parameters already set above, replace in the best from the grid search
# params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_depth'] = grid.best_params_['max_depth']
# #params['n_estimators'] = grid.best_params_['n_estimators']
# params['num_leaves'] = grid.best_params_['num_leaves']
# params['subsample_freq'] = grid.best_params_['subsample_freq']
# params['min_child_weight'] = grid.best_params_['min_child_weight']
# params['reg_alpha'] = grid.best_params_['reg_alpha']
# params['reg_lambda'] = grid.best_params_['reg_lambda']
# #params['subsample'] = grid.best_params_['subsample']
# # params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']
#
# print('Fitting with params: ')
# print(params)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 4500,
                 early_stopping_rounds= 10,
                 valid_sets= [train_data, valid_data],
                 verbose_eval= 10
                 )

#Predict on test set
predictions_lgbm_prob = lgbm.predict(test_df, num_iteration=lgbm.best_iteration)
predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
plt.hist(predictions_lgbm_prob)
print('Ratio:', predictions_lgbm_01.sum()/len(predictions_lgbm_01))

#--------------------------Print accuracy measures and variable importances----------------------
#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=40, importance_type='gain')

submission_df.is_click = predictions_lgbm_01
submission_df.to_csv('lgb_4_FE_2.csv', index = False)