#[824]	training's auc: 0.624502	valid_1's auc: 0.594755 PL 0.5471

#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer

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
train_df['hour'] = train_df['DateTime'].str.slice(11,13)
train_df['hour'] = train_df['hour'].astype(int)

#Aggregate historical data features
del historical_data_df['DateTime']
# hist_agg_unique = historical_data_df.groupby('user_id').nunique()[['product', 'action']]
# hist_agg_unique.columns = ['product_nunique', 'action_nunique']
#
# hist_agg_count = pd.DataFrame(historical_data_df.groupby('user_id').count()['product'])
# hist_agg_count.columns = ['visit_count']
#
# train_df = train_df.merge(right=hist_agg_unique.reset_index(), how='left', on='user_id')
# train_df = train_df.merge(right=hist_agg_count.reset_index(), how='left', on='user_id')



#Set as categorical the respectful variables
cat_cols = ['user_id', 'product', 'campaign_id', 'webpage_id', 'product_category_1',
            'product_category_2', 'user_group_id', 'gender', 'age_level',
            'user_depth', 'city_development_index', 'var_1', 'hour',
            ]

for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    train_df[col] = train_df[col].cat.codes

#Replace -1 with null
train_df[train_df==-1] = np.nan

#impute
#train_df['product_nunique'] = train_df['product_nunique'].fillna(train_df['product_nunique'].mean())
#train_df['action_nunique'] = train_df['action_nunique'].fillna(train_df['action_nunique'].mean())
#train_df['visit_count'] = train_df['visit_count'].fillna(train_df['visit_count'].mean())

#Drop unneeded features
train_df = train_df.drop(['session_id',
                          'campaign_id',
                      'DateTime',
                      'hour',
                      'gender',
                      'user_id',
                      'product_category_2',
                      'city_development_index',
                      #'visit_count'
                      ], axis=1)

#---------------------Preprocess Test set--------------------
print('test preprocess')
#Extract time
test_df['hour'] = test_df['DateTime'].str.slice(11,13)
test_df['hour'] = test_df['hour'].astype(int)

#Aggregate historical data features
#aggregations have been calculate in train set

#test_df = test_df.merge(right=hist_agg_unique.reset_index(), how='left', on='user_id')
#test_df = test_df.merge(right=hist_agg_count.reset_index(), how='left', on='user_id')

#Set as categorical the respectful variables
cat_cols = ['user_id', 'product', 'campaign_id', 'webpage_id', 'product_category_1',
            'product_category_2', 'user_group_id', 'gender', 'age_level',
            'user_depth',
            'city_development_index', 'var_1', 'hour',
            ]

for col in cat_cols:
    test_df[col] = test_df[col].astype('category')
    test_df[col] = test_df[col].cat.codes

#Replace -1 with null
test_df[test_df==-1] = np.nan

#impute
#test_df['product_nunique'] = test_df['product_nunique'].fillna(test_df['product_nunique'].mean())
#test_df['action_nunique'] = test_df['action_nunique'].fillna(test_df['action_nunique'].mean())
#test_df['visit_count'] = test_df['visit_count'].fillna(test_df['visit_count'].mean())

#Drop unneeded features
test_df = test_df.drop(['session_id',
                        'campaign_id',
                      'DateTime',
                      'hour',
                      'gender',
                      'user_id',
                      'product_category_2',
                      'city_development_index',
                      #'visit_count'
                      ], axis=1)

#Split in train and validation set
train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df, label_y, test_size=0.2, stratify=label_y)


#Impute missing
the_imputer = Imputer(strategy = 'most_frequent')
train_early_x = the_imputer.fit_transform(train_early_x)
valid_early_x = the_imputer.fit_transform(valid_early_x)
test_df = the_imputer.fit_transform(test_df)

#------------------------Build LightGBM Model-----------------------
regr = RandomForestClassifier(bootstrap= True, max_depth=6, n_estimators = 30, max_leaf_nodes = 21, min_impurity_split = 0.1, class_weight={1:14})
regr.fit(train_early_x, train_early_y)

print(regr.feature_importances_)

predictions_valid = regr.predict_proba(valid_early_x)
predictions_valid_01 = regr.predict(valid_early_x)
plt.hist(predictions_valid[:,1])
print('ratio:', predictions_valid_01.sum()/len(predictions_valid_01))

auc_valid = roc_auc_score(valid_early_y,predictions_valid_01)
print('AUC:', auc_valid)

# #Predict on test set
# predictions_lgbm_prob = lgbm.predict(test_df, num_iteration=lgbm.best_iteration)
# predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
# plt.hist(predictions_lgbm_prob)
# predictions_lgbm_01.sum()/len(predictions_lgbm_01)
#
# #--------------------------Print accuracy measures and variable importances----------------------
# #Plot Variable Importances
# lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')
#
# submission_df.is_click = predictions_lgbm_01
# submission_df.to_csv('lgb_3.csv', index = False)