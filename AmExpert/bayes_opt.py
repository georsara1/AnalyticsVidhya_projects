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
from bayes_opt import BayesianOptimization
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
train_df = train_df.drop(['session_id',
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

test_df = test_df.drop(['session_id',
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
    train_df[col] = train_df[col].astype('category')
    train_df[col] = train_df[col].cat.codes
    test_df[col] = test_df[col].astype('category')
    test_df[col] = test_df[col].cat.codes

# Replace -1 with null
train_df[train_df == -1] = np.nan
test_df[test_df == -1] = np.nan


# Frequency encoding
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size() / frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0: '{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')


len_train = train_df.shape[0]
df_all = pd.concat([train_df, test_df])

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

train_df = df_all[:len_train]
test_df = df_all[len_train:]

# #Create new numerical
# train_df['interest_log'] = np.log1p(train_df['interest'])
# train_df['popularity_log'] = np.log1p(train_df['popularity'])
#
# train_df['interest_squared'] = train_df['interest']**2
# train_df['popularity_squared'] = train_df['popularity']**2
#
# train_df['int2pop'] = train_df['interest'] / train_df['popularity']
# train_df['user2product'] = train_df['user_id_Frequency'] * train_df['product_Frequency']
#
# train_df['interest_log'] = np.log1p(test_df['interest'])
# test_df['popularity_log'] = np.log1p(test_df['popularity'])
#
# test_df['interest_squared'] = test_df['interest']**2
# test_df['popularity_squared'] = test_df['popularity']**2
#
# test_df['int2pop'] = test_df['interest'] / test_df['popularity']
# test_df['user2product'] = test_df['user_id_Frequency'] * test_df['product_Frequency']

# Split in train and validation set
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df, label_y, test_size=0.2,
                                                                              stratify=label_y)

cat_feat_lgb = ['user_id',
                #'product',
                'campaign_id',
                'webpage_id',
                #'product_category_1',
                'product_category_2',
                'user_group_id',
                'age_level',
                #'user_depth',
                'var_1',
                #'city_development_index'
                 ]

def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000,
                            learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature=cat_feat_lgb, free_raw_data=False)

    # parameters
    def lgb_eval(num_leaves, learning_rate, max_depth, lambda_l1, lambda_l2,
                 min_child_weight):
        params = {'application': 'binary', 'num_iterations': n_estimators, 'learning_rate': learning_rate,
                  'early_stopping_round': 100, 'metric': 'auc', 'is_unbalance' : 'True'}
        params["num_leaves"] = int(round(num_leaves))
        params['learning_rate'] = max(learning_rate, 0)
        #params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        #params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        #params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval=200,
                           metrics=['auc'])
        return max(cv_result['auc-mean'])

    # range
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (16, 30),
                                            'learning_rate': (0.01,0.06),
                                            #'feature_fraction': (0.1, 0.9),
                                            #'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            #'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (1, 50)
                                            },
                                             random_state=0
                                             )
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    # output optimization process
    if output_process == True: lgbBO.points_to_csv("bayes_opt_result.csv")

    # return best parameters
    return lgbBO.res['max']['max_params']


opt_params = bayes_parameter_opt_lgb(train_early_x, train_early_y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100,
                                     learning_rate=0.05)

opt_params["num_leaves"] = int(opt_params["num_leaves"])
#opt_params["feature_fraction"] = int(opt_params["feature_fraction"])+1
opt_params["max_depth"] = int(opt_params["max_depth"])

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : opt_params["max_depth"],
          'objective': 'binary',
          'nthread': 32,
          'n_estimators': 100,
          'num_leaves': opt_params["num_leaves"],
          'learning_rate': opt_params["learning_rate"],
          #'max_bin': 512,
          #'subsample_for_bin': 200,
          #'subsample': 0.7,
          #'subsample_freq': 5,
          #'colsample_bytree': 0.9,
          'reg_alpha': opt_params["lambda_l1"],
          'reg_lambda': opt_params["lambda_l2"],
          'min_child_weight': opt_params["min_child_weight"],
          #'min_child_samples': 5,
          #'scale_pos_weight': 1,
          #'num_class' : 2,
          'metric' : 'auc',
          'is_unbalance' : 'True'
          }

train_data=lgb.Dataset(train_early_x,label=train_early_y)
valid_data=lgb.Dataset(valid_early_x,label=valid_early_y)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 1444,
                 categorical_feature = cat_feat_lgb,
                 early_stopping_rounds= 10,
                 valid_sets= [train_data, valid_data],
                 verbose_eval= 10,

                 )

# Predict on test set
predictions_lgbm_prob = lgbm.predict(test_df, num_iteration=lgbm.best_iteration)
predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0)  # Turn probability to 0-1 binary output
plt.hist(predictions_lgbm_prob)
print('Ratio:', predictions_lgbm_01.sum() / len(predictions_lgbm_01))

# --------------------------Print accuracy measures and variable importances----------------------
# Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=21, importance_type='gain')

submission_df.is_click = predictions_lgbm_01
submission_df.to_csv('lgb_4_FE_bayes.csv', index=False)