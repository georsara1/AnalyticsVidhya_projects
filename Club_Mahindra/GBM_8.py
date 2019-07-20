# [250]	 PL 97.1104353082005
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import lightgbm as lgb
from math import sqrt, floor


#Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sample_submission.csv')

df_test['amount_spent_per_room_night_scaled'] = 0

#Concatenate into a single data frame
df_all = pd.concat([df_train, df_test], axis = 0)

df_all = df_all.rename(columns={'amount_spent_per_room_night_scaled': 'target'})

#Feature engineering and transformations
df_all['checkin_date'] = pd.to_datetime(df_all['checkin_date'], format = '%d/%m/%y')
df_all['checkout_date'] = pd.to_datetime(df_all['checkout_date'], format = '%d/%m/%y')
df_all['booking_date'] = pd.to_datetime(df_all['booking_date'], format = '%d/%m/%y')

df_all['book_days_in_advance'] = (df_all['checkin_date'] - df_all['booking_date']).dt.days
df_all['book_days_in_advance'][df_all['book_days_in_advance']<0] = np.nan

df_all['nights_stay'] = (df_all['checkout_date'] - df_all['checkin_date']).dt.days

df_all['diff_nights_roomnights'] = df_all['roomnights'] - df_all['nights_stay']
# df_all['diff_nights_roomnights'][df_all['diff_nights_roomnights']<0] = np.nan
# df_all['diff_nights_roomnights'].fillna(df_all['diff_nights_roomnights'].median())

#df_all['people_per_room'] =  df_all['total_pax'] / df_all['nights_stay']

df_all['checkin_month'] = df_all['checkin_date'].dt.month
df_all['checkin_year'] = df_all['checkin_date'].dt.year
df_all['checkin_dow'] = df_all['checkin_date'].dt.dayofweek
df_all['checkin_day'] = df_all['checkin_date'].dt.day
#df_all['checkin_season'] = (df_all['checkin_month']%12 + 3)//3

#df_all['checkout_month'] = df_all['checkout_date'].dt.month
#df_all['checkin_year'] = df_all['checkin_date'].dt.year
#df_all['checkout_dow'] = df_all['checkout_date'].dt.dayofweek
#df_all['checkout_day'] = df_all['checkout_date'].dt.day

#df_all['is_same_state_vacation'] = df_all.apply(lambda x:1 if x['state_code_residence']==x['state_code_resort'] else 0, axis =1)
#df_all['state_resort_interaction'] = df_all['resort_region_code'].map(str) + df_all['resort_type_code'].map(str)

df_all['roomnights'] = np.abs(df_all['roomnights'])

df_all['checkin_year'][df_all['checkin_year']== 2012] = 2018

# Fixing missing value in 'state_code_residence' Column
df_all['state_code_residence'] = df_all['state_code_residence'].fillna('Unidentified')
# df_all['state_code_residence_NA'] = np.where(df_all['state_code_residence'] == 'Unidentified'
#                                     ,True
#                                     ,False
#                                     )


#Categorical encoding
cat_cols = ['channel_code', 'main_product_code', 'persontravellingid', 'resort_region_code',
       'resort_type_code', 'room_type_booked_code',
       'season_holidayed_code', 'state_code_residence', 'state_code_resort',
       'member_age_buckets', 'booking_type_code',
       'cluster_code',  'resort_id'
       ]

for col in cat_cols:
    df_all[col] = df_all[col].astype('category')
    df_all[col] = df_all[col].cat.codes


#Mean encoding
train = df_all.iloc[:df_train.shape[0],:]

def mean_encode(target, column_name, dataframe, name):
    df = dataframe[target].groupby(dataframe[column_name]).agg({ name + '_mean':'mean'})
    df.reset_index(inplace=True)
    return df

resort_type_code_df = mean_encode('target', 'resort_region_code', train, 'persontravellingid_mean_enc')
df_all  = pd.merge(df_all, resort_type_code_df, how='left', on='resort_region_code')
room_type_booked_code_df = mean_encode('target', 'room_type_booked_code', train, 'room_type_booked_code_mean_enc')
df_all  = pd.merge(df_all, room_type_booked_code_df, how='left', on='room_type_booked_code')


# #Drop unneeded
# df_all = df_all.drop(['reservation_id', 'booking_date', 'checkin_date', 'checkout_date', 'memberid',
# 'reservationstatusid_code',
#                        ], axis = 1)

#Split in train, validation and test sets
train = df_all.iloc[:df_train.shape[0],:]
test = df_all.iloc[df_train.shape[0]:,:]

memberid_train_counts = pd.DataFrame(df_train['memberid'].value_counts()).reset_index()
memberid_test_counts = pd.DataFrame(df_test['memberid'].value_counts()).reset_index()

memberid_train_counts.columns = ['memberid', 'memberid_no_of_visits']
memberid_test_counts.columns = ['memberid', 'memberid_no_of_visits']

train  = pd.merge(train, memberid_train_counts, how='left', on='memberid')
test  = pd.merge(test, memberid_test_counts, how='left', on='memberid')

# #How many products each person has visited
# memberid_resort_counts_train = pd.DataFrame(train.groupby(['memberid'])['resort_id'].count()).reset_index()
# memberid_resort_counts_test = pd.DataFrame(test.groupby(['memberid'])['resort_id'].count()).reset_index()
#
# memberid_resort_counts_train.columns = ['memberid', 'memberid_no_of_hotels']
# memberid_resort_counts_test.columns = ['memberid', 'memberid_no_of_hotels']
#
# train  = pd.merge(train, memberid_resort_counts_train, how='left', on='memberid')
# test  = pd.merge(test, memberid_resort_counts_test, how='left', on='memberid')

#How many nights each person has booked
memberid_nights_train = pd.DataFrame(train.groupby(['memberid'])['nights_stay'].mean()).reset_index()
memberid_nights_test = pd.DataFrame(test.groupby(['memberid'])['nights_stay'].mean()).reset_index()

memberid_nights_train.columns = ['memberid', 'memberid_no_of_nights_mean']
memberid_nights_test.columns = ['memberid', 'memberid_no_of_nights_mean']

train  = pd.merge(train, memberid_nights_train, how='left', on='memberid')
test  = pd.merge(test, memberid_nights_test, how='left', on='memberid')

#How many roomnights each person has booked
memberid_roomnights_train = pd.DataFrame(train.groupby(['memberid'])['roomnights'].mean()).reset_index()
memberid_roomnights_test = pd.DataFrame(test.groupby(['memberid'])['roomnights'].mean()).reset_index()

memberid_roomnights_train.columns = ['memberid', 'memberid_no_of_nightrooms']
memberid_roomnights_test.columns = ['memberid', 'memberid_no_of_nightrooms']

train  = pd.merge(train, memberid_roomnights_train, how='left', on='memberid')
test  = pd.merge(test, memberid_roomnights_test, how='left', on='memberid')

# #Earliest booking a person has made
# memberid_mindate_train = pd.DataFrame(train.groupby(['memberid'])['checkin_date'].max()).reset_index()
# memberid_mindate_test = pd.DataFrame(test.groupby(['memberid'])['checkin_date'].max()).reset_index()
#
# memberid_mindate_train.columns = ['memberid', 'days_since_first_booking']
# memberid_mindate_test.columns = ['memberid', 'days_since_first_booking']
#
# memberid_mindate_train['days_since_first_booking'] = -(memberid_mindate_train['days_since_first_booking'] - train['checkin_date'].min()).dt.days
# memberid_mindate_test['days_since_first_booking'] = -(memberid_mindate_test['days_since_first_booking'] - test['checkin_date'].min()).dt.days
#
# train  = pd.merge(train, memberid_mindate_train, how='left', on='memberid')
# test  = pd.merge(test, memberid_mindate_train, how='left', on='memberid')




# #Drop unneeded
train = train.drop(['reservation_id', 'booking_date', 'checkin_date', 'checkout_date', 'memberid',
'reservationstatusid_code',
                       ], axis = 1)
test = test.drop(['reservation_id', 'booking_date', 'checkin_date', 'checkout_date', 'memberid',
'reservationstatusid_code',
                       ], axis = 1)


train_early_stop_x, valid_early_stop_x = train_test_split(train, test_size= 0.15, random_state= 7)

train_early_stop_y = train_early_stop_x['target']
del train_early_stop_x['target']

valid_early_stop_y = valid_early_stop_x['target']
del valid_early_stop_x['target']

del test['target']


#Build model
train_data=lgb.Dataset(train_early_stop_x,label=train_early_stop_y)
valid_data = lgb.Dataset(valid_early_stop_x, valid_early_stop_y, reference=train_data)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 7,
          'objective': 'regression',
          #'nthread': 5,
          'num_leaves': 11,
          'learning_rate': 0.08,
          #'max_bin': 512,
          #'subsample_for_bin': 200,
          #'subsample': 1,
          #'subsample_freq': 1,
          'colsample_bytree': 0.8,
'min_data_in_leaf': 40,
          'reg_alpha': 1,
          'reg_lambda': 1,
          #'min_split_gain': 0.5,
          #'min_child_weight': 1,
          #'min_child_samples': 5,
          #'scale_pos_weight': 1,
          #'num_class' : 1,
          'metric' : 'rmse'
          }

metrics_lgb = {}

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 250,
                 valid_sets=[train_data, valid_data],
                 evals_result= metrics_lgb,
                 categorical_feature=cat_cols,
                 early_stopping_rounds= 20,
                 verbose_eval= 50
                 )

#Plot training
ax = lgb.plot_metric(metrics_lgb, metric = 'rmse')

#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=30, importance_type='gain')

#Predict on test set
predictions = lgbm.predict(valid_early_stop_x)
#predictions = [floor(p) if p>=0 else 0 for p in predictions]

valid_rmse = sqrt(mean_squared_error(valid_early_stop_y, predictions))
print('Validation RMSE: {}'.format(valid_rmse))

#Predict on test set
predictions = lgbm.predict(test)
#predictions = [floor(p) if p>=0 else 0 for p in predictions]

df_submission['amount_spent_per_room_night_scaled'] = predictions

df_submission.to_csv('GBM_8.csv', index = False)
