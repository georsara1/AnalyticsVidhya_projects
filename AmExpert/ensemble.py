# [144]	training's auc: 0.644202	valid_1's auc: 0.606775 PL 0.5690
# [144]	training's auc: 0.643946	valid_1's auc: 0.615326 PL 0.5704
# [144]	training's auc: 0.64195	    valid_1's auc: 0.619794 PL 0.5716

# Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sns.set_style("whitegrid")

# Import data
print('import data')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
historical_data_df = pd.read_csv('historical_user_logs.csv')
submission_df = pd.read_csv('sample_submission.csv')

label_y = train_df['is_click']
del train_df['is_click']

# ---------------------Preprocess Train set--------------------
print('train preprocess')
# Extract time
# train_df['hour'] = train_df['DateTime'].str.slice(11,13)
# train_df['hour'] = train_df['hour'].astype(int)

# Aggregate historical data features
del historical_data_df['DateTime']
# hist_agg_unique = historical_data_df.groupby('user_id').nunique()[['product', 'action']]
# hist_agg_unique.columns = ['product_nunique', 'action_nunique']
#
hist_agg_count = pd.DataFrame(historical_data_df.groupby('product').count())
# hist_agg_count.columns = ['visit_count']
hist_agg_count['popularity'] = hist_agg_count['action'] / hist_agg_count['action'].sum()
hist_agg_count = hist_agg_count.drop(['user_id', 'action'], axis=1)

hist_agg_count_users = pd.DataFrame(historical_data_df.groupby('user_id').count())
del hist_agg_count_users['product']
hist_agg_count_users.columns = ['user_visits']

prod_interest = pd.crosstab(historical_data_df['product'], historical_data_df['action'], normalize='index')
prod_interest['interest'] = prod_interest['interest'] * 100
del prod_interest['view']

train_df = train_df.merge(right=prod_interest.reset_index(), how='left', on='product')
train_df = train_df.merge(right=hist_agg_count.reset_index(), how='left', on='product')
# train_df = train_df.merge(right=hist_agg_count_users.reset_index(), how='left', on='user_id')


# ---------------------Preprocess Test set--------------------
print('test preprocess')
# Extract time
# test_df['hour'] = test_df['DateTime'].str.slice(11,13)
# test_df['hour'] = test_df['hour'].astype(int)

# Aggregate historical data features
# aggregations have been calculate in train set

# test_df = test_df.merge(right=hist_agg_unique.reset_index(), how='left', on='user_id')
# test_df = test_df.merge(right=hist_agg_count.reset_index(), how='left', on='user_id')

test_df = test_df.merge(right=prod_interest.reset_index(), how='left', on='product')
test_df = test_df.merge(right=hist_agg_count.reset_index(), how='left', on='product')
# test_df = test_df.merge(right=hist_agg_count_users.reset_index(), how='left', on='user_id')


# Drop unneeded features
train_df1 = train_df.drop(['session_id',
                          # 'campaign_id',
                          'DateTime',
                          # 'hour',
                          'gender',
                          # 'user_id',
                          # 'product_category_2',
                          # 'city_development_index',
                          # 'age_level',
                          # 'webpage_id'
                          # 'visit_count'
                          ], axis=1)

test_df1 = test_df.drop(['session_id',
                        # 'campaign_id',
                        'DateTime',
                        # 'hour',
                        'gender',
                        # 'user_id',
                        # 'product_category_2',
                        # 'city_development_index',
                        # 'age_level',
                        # 'webpage_id'
                        # 'visit_count'
                        ], axis=1)

# Encode categorical
cat_cols = ['user_id',
            'product', 'campaign_id', 'webpage_id', 'product_category_1',
            'product_category_2',
            'user_group_id',
            # 'gender',
            'age_level',
            'user_depth',
            'city_development_index',
            'var_1',
            # 'hour',
            ]

# encode categorical
for col in cat_cols:
    train_df1[col] = train_df1[col].astype('category')
    train_df1[col] = train_df1[col].cat.codes
    test_df1[col] = test_df1[col].astype('category')
    test_df1[col] = test_df1[col].cat.codes

# Replace -1 with null
train_df1[train_df1 == -1] = np.nan
test_df1[test_df1 == -1] = np.nan


# Frequency encoding
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size() / frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0: '{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')


len_train = train_df1.shape[0]
df_all = pd.concat([train_df1, test_df1])

freq_cols = [  # 'product',
    'campaign_id',
    'webpage_id',
    'product_category_1',
    'product_category_2',
    'user_group_id',
    # 'gender',
    'age_level',
    'user_depth',
    # 'city_development_index',
    'var_1',
    # 'hour',
]

for col in tqdm(freq_cols):
    df_all = frequency_encoding(df_all, col)

train_df1 = df_all[:len_train]
test_df1 = df_all[len_train:]

# Split in train and validation set
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df1, label_y, test_size=0.2,
                                                                              stratify=label_y)

# ------------------------Build LightGBM Model-----------------------
train_data = lgb.Dataset(train_early_x, label=train_early_y)
valid_data = lgb.Dataset(valid_early_x, label=valid_early_y)

# Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth': 8,
          'objective': 'binary',
          'nthread': 32,
          'n_estimators': 144,
          'num_leaves': 21,
          'learning_rate': 0.02,
          # 'max_bin': 512,
          # 'subsample_for_bin': 200,
          # 'subsample': 0.7,
          # 'subsample_freq': 5,
          # 'colsample_bytree': 0.9,
          'reg_alpha': 3.11,
          'reg_lambda': 3.1,
          'min_child_weight': 0.9,
          # 'min_child_samples': 5,
          # 'scale_pos_weight': 1,
          # 'num_class' : 2,
          'metric': 'auc',
          'is_unbalance': 'True'
          }


cat_feat_lgb = ['user_id',
                # 'product',
                'campaign_id',
                'webpage_id',
                # 'product_category_1',
                'product_category_2',
                'user_group_id',
                'age_level',
                # 'user_depth',
                'var_1',
                # 'city_development_index'
                ]

# Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 4500,
                 categorical_feature=cat_feat_lgb,
                 early_stopping_rounds=10,
                 valid_sets=[train_data, valid_data],
                 verbose_eval=10
                 )

# Predict on test set
predictions_lgbm_prob_1 = lgbm.predict(test_df1, num_iteration=lgbm.best_iteration)
# predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0)  # Turn probability to 0-1 binary output
# plt.hist(predictions_lgbm_prob)
# print('Ratio:', predictions_lgbm_01.sum() / len(predictions_lgbm_01))

# --------------------------Print accuracy measures and variable importances----------------------
# Plot Variable Importances
#lgb.plot_importance(lgbm, max_num_features=21, importance_type='gain')

# Drop unneeded features
train_df2 = train_df.drop(['session_id',
                          # 'campaign_id',
                          'DateTime',
                          # 'hour',
                          'gender',
                          'user_id',
                          # 'product_category_2',
                          # 'city_development_index',
                          # 'age_level',
                          # 'webpage_id'
                          # 'visit_count'
                          ], axis=1)

test_df2 = test_df.drop(['session_id',
                        # 'campaign_id',
                        'DateTime',
                        # 'hour',
                        'gender',
                        'user_id',
                        # 'product_category_2',
                        # 'city_development_index',
                        # 'age_level',
                        # 'webpage_id'
                        # 'visit_count'
                        ], axis=1)

# Encode categorical
cat_cols = [#'user_id',
            'product', 'campaign_id', 'webpage_id', 'product_category_1',
            'product_category_2',
            'user_group_id',
            # 'gender',
            'age_level',
            'user_depth',
            'city_development_index',
            'var_1',
            # 'hour',
            ]

# encode categorical
for col in cat_cols:
    train_df2[col] = train_df2[col].astype('category')
    train_df2[col] = train_df2[col].cat.codes
    test_df2[col] = test_df2[col].astype('category')
    test_df2[col] = test_df2[col].cat.codes

# Replace -1 with null
train_df2[train_df2 == -1] = np.nan
test_df2[test_df2 == -1] = np.nan


# Frequency encoding
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size() / frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0: '{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')


len_train = train_df2.shape[0]
df_all = pd.concat([train_df2, test_df2])

freq_cols = [  # 'product',
    'campaign_id',
    'webpage_id',
    'product_category_1',
    'product_category_2',
    'user_group_id',
    # 'gender',
    'age_level',
    'user_depth',
    # 'city_development_index',
    'var_1',
    # 'hour',
]

for col in tqdm(freq_cols):
    df_all = frequency_encoding(df_all, col)

train_df2 = df_all[:len_train]
test_df2 = df_all[len_train:]

# Split in train and validation set
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df2, label_y, test_size=0.2,
                                                                              stratify=label_y)

# ------------------------Build LightGBM Model-----------------------
train_data = lgb.Dataset(train_early_x, label=train_early_y)
valid_data = lgb.Dataset(valid_early_x, label=valid_early_y)

# Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth': 8,
          'objective': 'binary',
          'nthread': 32,
          'n_estimators': 144,
          'num_leaves': 21,
          'learning_rate': 0.02,
          # 'max_bin': 512,
          # 'subsample_for_bin': 200,
          # 'subsample': 0.7,
          # 'subsample_freq': 5,
          # 'colsample_bytree': 0.9,
          'reg_alpha': 3.11,
          'reg_lambda': 3.1,
          'min_child_weight': 0.9,
          # 'min_child_samples': 5,
          # 'scale_pos_weight': 1,
          # 'num_class' : 2,
          'metric': 'auc',
          'is_unbalance': 'True'
          }

cat_feat_lgb = [#'user_id',
                # 'product',
                'campaign_id',
                'webpage_id',
                # 'product_category_1',
                'product_category_2',
                'user_group_id',
                'age_level',
                # 'user_depth',
                'var_1',
                # 'city_development_index'
                ]

# Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 4500,
                 categorical_feature=cat_feat_lgb,
                 early_stopping_rounds=10,
                 valid_sets=[train_data, valid_data],
                 verbose_eval=10
                 )

# Predict on test set
predictions_lgbm_prob_2 = lgbm.predict(test_df2, num_iteration=lgbm.best_iteration)

# predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0)  # Turn probability to 0-1 binary output
# plt.hist(predictions_lgbm_prob)
# print('Ratio:', predictions_lgbm_01.sum() / len(predictions_lgbm_01))

# --------------------------Print accuracy measures and variable importances----------------------
# Plot Variable Importances
#lgb.plot_importance(lgbm, max_num_features=21, importance_type='gain')
pred_ens = 0.6*predictions_lgbm_prob_1+0.5*predictions_lgbm_prob_2
predictions_lgbm_01 = np.where(pred_ens > 0.5, 1, 0)  # Turn probability to 0-1 binary output
plt.hist(pred_ens)
plt.hist(predictions_lgbm_prob_1)
plt.hist(predictions_lgbm_prob_2)
submission_df.is_click = pred_ens
submission_df.to_csv('lgb_4_FE_ens.csv', index=False)