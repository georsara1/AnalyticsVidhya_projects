
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import lightgbm as lgb
from math import sqrt, floor
from catboost import CatBoostRegressor, Pool, cv

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

df_all['diff_nights_roomnights'] = df_all['roomnights']- df_all['nights_stay']

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
df_all['state_code_residence_NA'] = np.where(df_all['state_code_residence'] == 'Unidentified'
                                    ,True
                                    ,False
                                    )


#Categorical encoding
cat_cols = ['channel_code', 'main_product_code', 'persontravellingid', 'resort_region_code',
       'resort_type_code', 'room_type_booked_code',
       'season_holidayed_code', 'state_code_residence', 'state_code_resort',
       'member_age_buckets', 'booking_type_code',
       'cluster_code', 'reservationstatusid_code', 'resort_id'
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

# Create correlation matrix
corr_matrix = df_all._get_numeric_data().corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.70
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

#Drop unneeded
df_all = df_all.drop(['reservation_id', 'booking_date', 'checkin_date', 'checkout_date', 'memberid'
                       ], axis = 1)

#Split in train, validation and test sets
train = df_all.iloc[:df_train.shape[0],:]
test = df_all.iloc[df_train.shape[0]:,:]

train_early_stop_x, valid_early_stop_x = train_test_split(train, test_size= 0.15, random_state= 7)

train_early_stop_y = train_early_stop_x['target']
del train_early_stop_x['target']

valid_early_stop_y = valid_early_stop_x['target']
del valid_early_stop_x['target']

del test['target']

print("\nCatBoost...")
train_pool = Pool(data=train_early_stop_x, label=train_early_stop_y)
validation_pool = Pool(data=valid_early_stop_x, label=valid_early_stop_y)

CB_model = CatBoostRegressor(n_estimators=1000,
                              verbose=100,
                              #custom_loss=['AUC', 'Accuracy'],
                              eval_metric='RMSE:hints=skip_train~false',
                              depth=7,
                              learning_rate=0.0618,
                              #l2_leaf_reg=4,
                              od_type="Iter",
                              od_wait=20,
                              use_best_model=True,
                              rsm=0.5
                              )

CB_model.fit(train_pool, eval_set=validation_pool, plot=True)



#Predict on test set
y_preds = CB_model.predict(valid_early_stop_x)
#predictions = [floor(p) if p>=0 else 0 for p in predictions]

valid_rmse = sqrt(mean_squared_error(valid_early_stop_y, y_preds))
print('Validation RMSE: {}'.format(valid_rmse))

#Predict on test set
y_preds = CB_model.predict(test)
#predictions = [floor(p) if p>=0 else 0 for p in predictions]

df_submission['amount_spent_per_room_night_scaled'] = y_preds

df_submission.to_csv('cat_2.csv', index = False)



