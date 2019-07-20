# Results of this script:
# [420]	valid_0's auc: 0.782757
# Public leaderboard 0.782
# submitted as test-komo1
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
le = LabelEncoder()
from tqdm import tqdm

print('Importing data...')
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
meal_info = pd.read_csv('meal_info.csv')
center_info = pd.read_csv('fulfilment_center_info.csv')
lgbm_submission = pd.read_csv('sample_submission.csv')

#Separate target variable
y = np.log1p(data['num_orders'])
del data['num_orders']

#Pre-process train set
all_data = data.merge(right=meal_info, how='left', on='meal_id')
all_data = all_data.merge(right=center_info, how='left', on='center_id')

all_data['discount'] = (all_data['base_price'] - all_data['checkout_price']) / all_data['base_price']
all_data['posneg'] = all_data.apply(lambda x: 0 if x['discount']>=0 else 1, axis = 1)
all_data['price_range'] = all_data.apply(lambda x: 'low' if x['base_price']<=210 else ('mid' if x['base_price']<=380 else 'high'), axis = 1)

del all_data['id']
del all_data['center_id']
del all_data['meal_id']
del all_data['week']


#Pre-process test set
test = test.merge(right=meal_info, how='left', on='meal_id')
test = test.merge(right=center_info, how='left', on='center_id')

test['discount'] = (test['base_price'] - test['checkout_price']) / test['base_price']
test['posneg'] = test.apply(lambda x: 0 if x['discount']>=0 else 1, axis = 1)
test['price_range'] = test.apply(lambda x: 'low' if x['base_price']<=210 else ('mid' if x['base_price']<=380 else 'high'), axis = 1)

del test['id']
del test['center_id']
del test['meal_id']
del test['week']

#Categorical variables
cat_cols = ['emailer_for_promotion',
            'homepage_featured',
            'category',
            'cuisine',
            'city_code',
            'region_code',
            'center_type',
            'posneg',
            'price_range'
            ]

for col in cat_cols:
    all_data[col] = all_data[col].astype('category')
    test[col] = test[col].astype('category')
    all_data[col] = all_data[col].cat.codes
    test[col] = test[col].cat.codes
    # all_data[col] = pd.factorize(all_data[col])
    # test[col] = pd.factorize(test[col])


# #Frequency encoding
#Categorical variables
freq_cols = ['emailer_for_promotion',
            'homepage_featured',
            'category',
            'cuisine',
            'city_code',
            'region_code',
            'center_type',
            'posneg',
             'price_range'
            ]

def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

len_train = all_data.shape[0]
df_all = pd.concat([all_data, test])

for col in tqdm(freq_cols):
    df_all = frequency_encoding(df_all, col)

train_df = df_all[:len_train]
test_df = df_all[len_train:]
# train_df = all_data.copy()
# test_df = test.copy()

#Modelling with LightGBM
oof_pred = np.zeros((test_df.shape[0], 10))

for i in range(10):
    #Split in train and validation set
    train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df, y, test_size=0.2)


    #------------------------Build LightGBM Model-----------------------
    train_data=lgb.Dataset(train_early_x,label=train_early_y)
    valid_data=lgb.Dataset(valid_early_x,label=valid_early_y)

    #Select Hyper-Parameters
    params = {'boosting_type': 'gbdt',
              'max_depth' : 8,
              'objective': 'regression',
              'nthread': 32,
              'n_estimators': 800,
              'num_leaves': 26,
              'learning_rate': 0.1,
              #'max_bin': 512,
              #'subsample_for_bin': 200,
              'subsample': 0.8,
              #'subsample_freq': 5,
              'colsample_bytree': 0.9,
              'reg_alpha': 5.21,
              'reg_lambda': 2.1,
              'min_child_weight': 4,
              #'min_child_samples': 35,
              #"bagging_fraction": 0.7,
              #"feature_fraction": 0.5,
              #"bagging_frequency": 5,
              #'scale_pos_weight': 1,
              #'num_class' : 2,
              'metric' : 'rmse',
              #'is_unbalance' : 'True'
              }

    cat_gbm = ['emailer_for_promotion',
                'homepage_featured',
                'category',
                'cuisine',
                'city_code',
                'region_code',
                'center_type',
                'posneg',
               'price_range'
                ]

    #Train model on selected parameters and number of iterations
    lgbm = lgb.train(params,
                     train_data,
                     4500,
                     categorical_feature= cat_gbm,
                     early_stopping_rounds= 20,
                     valid_sets= [train_data, valid_data],
                     verbose_eval= 10
                     )

    #Predict on test set
    predictions_lgbm = lgbm.predict(test_df, num_iteration=lgbm.best_iteration)
    #plt.hist(predictions_lgbm)
    #plt.show()
    oof_pred[:,i] = predictions_lgbm

final_pred = oof_pred.mean(axis = 1)
final_pred[final_pred<0] = 0

#--------------------------Print accuracy measures and variable importances----------------------
#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=27, importance_type='gain')
plt.tight_layout()
plt.show()

lgbm_submission.num_orders = np.expm1(final_pred)
lgbm_submission.to_csv('lgb_3.csv', index = False)