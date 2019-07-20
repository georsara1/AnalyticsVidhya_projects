import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, mean_squared_error
sns.set_style("whitegrid")
from statsmodels.tsa.ar_model import AR

volume_forecasts = pd.read_csv('volume_forecast_test.csv')
sku_recommendations = pd.read_csv('sku_recommendation_test.csv')
demographics = pd.read_csv('demographics.csv')
event_calendar = pd.read_csv('event_calendar.csv')
historical_volume = pd.read_csv('historical_volume.csv')
industry_volume = pd.read_csv('industry_volume.csv')
price_sales_promotion = pd.read_csv('price_sales_promotion.csv')
weather = pd.read_csv('weather.csv')
df_test = pd.read_csv('volume_forecast_test.csv')

#----------------------------Build Train set------------------------------
#Join historical volume with respective weather temperatures
all_data = pd.merge(historical_volume, weather,  how='left',
                  left_on=['Agency','YearMonth'], right_on = ['Agency','YearMonth'])

#Join all data with demographics
all_data = pd.merge(all_data, demographics,  how='left',
                  left_on=['Agency'], right_on = ['Agency'])

#Join all data with industry volume
all_data = pd.merge(all_data, industry_volume,  how='left',
                  left_on=['YearMonth'], right_on = ['YearMonth'])

#Join all data volume with price sales promotion data
all_data = pd.merge(all_data, price_sales_promotion,  how='left',
                  left_on=['Agency','SKU','YearMonth'], right_on = ['Agency','SKU','YearMonth'])

#Join all data with event calendar
# all_data = pd.merge(all_data, event_calendar,  how='left',
#                   left_on=['YearMonth'], right_on = ['YearMonth'])

df_train = all_data.drop(['YearMonth'], axis = 1)

df_train.Price[df_train.Price == 0] = df_train.Price.median()
df_train.Sales[df_train.Sales == 0] = df_train.Sales.median()
df_train.Promotions[df_train.Promotions == 0] = df_train.Promotions.median()


#-------------------------------Build Test set------------------------------
#1. For weather dataset
weather['YearMonth'] = weather.YearMonth.astype(str)
weather['Year'] = weather.YearMonth.str[0:4]
weather['Month'] = weather.YearMonth.str[4:6]
weather = weather.drop(['YearMonth'], axis = 1)
weather['Month'] = weather.Month.astype(str)
weather_jan = weather[weather.Month == '01']
Avg_Temps = weather_jan.groupby(['Agency'], as_index=False)['Avg_Max_Temp'].mean()

df_test = pd.merge(df_test, Avg_Temps,  how='left',
                  left_on=['Agency'], right_on = ['Agency'])

#2. For Average population and House income
df_test = pd.merge(df_test, demographics,  how='left',
                  left_on=['Agency'], right_on = ['Agency'])

#3. For Industry Volume
industry_volume['YearMonth'] = industry_volume.YearMonth.astype(str)
industry_volume['Year'] = industry_volume.YearMonth.str[0:4]
industry_volume['Month'] = industry_volume.YearMonth.str[4:6]
industry_volume = industry_volume.drop(['YearMonth'], axis = 1)
industry_volume['Month'] = industry_volume.Month.astype(str)
industry_volume_Jan = industry_volume[industry_volume.Month == '01']
industry_volume_Jan['Year'] = industry_volume.Year.astype(int)

df_test['Industry_Volume'] = 628000000

#4. For Price, Sales and Promotions
price_sales_promotion['YearMonth'] = price_sales_promotion.YearMonth.astype(str)
price_sales_promotion['Year'] = price_sales_promotion.YearMonth.str[0:4]
price_sales_promotion['Month'] = price_sales_promotion.YearMonth.str[4:6]
price_sales_promotion = price_sales_promotion.drop(['YearMonth'], axis = 1)
price_sales_promotion['Month'] = price_sales_promotion.Month.astype(str)
price_sales_promotion_Jan = price_sales_promotion[price_sales_promotion.Month == '01']
Max_Price = price_sales_promotion_Jan.groupby(['SKU'], as_index=False)['Price'].mean()
Avg_Sales = price_sales_promotion_Jan.groupby(['SKU'], as_index=False)['Sales'].mean()
Avg_Promotions = price_sales_promotion_Jan.groupby(['SKU'], as_index=False)['Promotions'].mean()

df_test = pd.merge(df_test, Max_Price,  how='left',
                  left_on=['SKU'], right_on = ['SKU'])

df_test = pd.merge(df_test, Avg_Sales,  how='left',
                  left_on=['SKU'], right_on = ['SKU'])

df_test = pd.merge(df_test, Avg_Promotions,  how='left',
                  left_on=['SKU'], right_on = ['SKU'])

#5. For Events
# event_calendar['YearMonth'] = event_calendar.YearMonth.astype(str)
# event_calendar['Year'] = event_calendar.YearMonth.str[0:4]
# event_calendar['Month'] = event_calendar.YearMonth.str[4:6]
# event_calendar = event_calendar.drop(['YearMonth'], axis = 1)
# event_calendar['Month'] = event_calendar.Month.astype(str)
# event_calendar = event_calendar[event_calendar.Month == '01']
#
# df_test['Easter Day'] = np.zeros(df_test.shape[0])
# df_test['Good Friday'] = np.zeros(df_test.shape[0])
# df_test['New Year'] = np.ones(df_test.shape[0])
# df_test['Christmas'] = np.zeros(df_test.shape[0])
# df_test['Labor Day'] = np.zeros(df_test.shape[0])
# df_test['Independence Day'] = np.zeros(df_test.shape[0])
# df_test['Revolution Day Memorial'] = np.zeros(df_test.shape[0])
# df_test['Regional Games'] = np.zeros(df_test.shape[0])
# df_test['FIFA U-17 World Cup'] = np.zeros(df_test.shape[0])
# df_test['Football Gold Cup'] = np.zeros(df_test.shape[0])
# df_test['Beer Capital'] = np.zeros(df_test.shape[0])
# df_test['Music Fest'] = np.zeros(df_test.shape[0])

#Add the Month and Year Variables
#df_test['Month'] = '01'
#df_test['Month'] = df_test.Month.astype('category')
#df_test['Year'] = int(2018)

#Set correct feature types
df_test['Agency'] = df_test.Agency.astype('category')
df_test['SKU'] = df_test.SKU.astype('category')

#Drop target variable (Volume)
df_test = df_test.drop(['Volume'], axis = 1)


#---------------------------Pre-processing-------------------------

#Create two new variables: Year and Month
# df_train['YearMonth'] = df_train.YearMonth.astype(str)
# df_train['Month'] = df_train.YearMonth.str[4:6]
# df_train['Year'] = df_train.YearMonth.str[0:4]


#Set correct feature types
# df_train['Year'] = df_train.Year.astype(int)
# df_train['Month'] = df_train.Month.astype('category')
df_train['Agency'] = df_train.Agency.astype('category')
df_train['SKU'] = df_train.SKU.astype('category')


#Encode categorical variables to ONE-HOT
# print('Converting categorical variables to numeric...')
#
# categorical_columns = ['Agency', 'SKU']
#
# df_train = pd.get_dummies(df_train, columns = categorical_columns,
#                     #drop_first = True #Slightly better performance with n columns in One-Hot encoding
#                     )

#Scaling slightly worsened the results in Gradient Boosting (kept in comments below for reference purposes)
#
# #Scale variables to [0,1] range
# columns_to_scale = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT5'
#     , 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
#
# df[columns_to_scale]=df[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))


#Split in 75% train and 25% test set
train, dev = train_test_split(df_train, test_size = 0.25, random_state= 1984)

train_y = train.Volume
dev_y = dev.Volume

train_x = train.drop(['Volume'], axis = 1)
dev_x = dev.drop(['Volume'], axis = 1)

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data = lgb.Dataset(dev_x, label= dev_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 4,
          'objective': 'regression',
          'nthread': 5,
          'num_leaves': 16,
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
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'root_mean_squared_error'
          }

# Create parameters to search
gridParams = {
    'learning_rate': [0.05],
    'n_estimators': [8,16],
    'num_leaves': [16, 20, 24],
    'boosting_type' : ['gbdt'],
    'objective' : ['regression'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.64, 0.65],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1, 1.2],
    'reg_lambda' : [ 1.2, 1.4],
    }

# Create classifier to use. Note that parameters have to be input manually, not as a dict!
mdl = lgb.LGBMRegressor(boosting_type= 'gbdt',
          objective = 'regression',
          n_jobs = 5,
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight']
                         )

# To view the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)

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
                 18000,
                 early_stopping_rounds= 40,
                 valid_sets= [valid_data],
                 verbose_eval= 4
                 )

ax = lgb.plot_importance(lgbm, max_num_features=21, importance_type= 'gain')
plt.show()

#Then we must re-train our model on the entire dataset
train_y = df_train.Volume
train_x = df_train.drop(['Volume'], axis = 1)

train_data=lgb.Dataset(train_x,label=train_y)

params = {'boosting_type': 'gbdt',
          'max_depth' : 4,
          'objective': 'regression',
          'nthread': 5,
          'num_leaves': 31,
          #'n_estimators': 31,
          'learning_rate': 0.08,
          'max_bin': 256,
          'subsample_for_bin': 200,
          'subsample': 0.7,
          'subsample_freq': 1,
          'colsample_bytree': 0.64,
          'reg_alpha': 4,
          'reg_lambda': 1.4,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'mean_squared_error'
          }

lgbm = lgb.train(params,
                 train_data,
                 3500,
                 #early_stopping_rounds= 40,
                 #valid_sets= [valid_data],
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm = lgbm.predict(df_test)

volume_forecasts.Volume = predictions_lgbm
plt.hist(volume_forecasts.Volume, bins = 100)
volume_forecasts.Volume.min()

volume_forecasts.Volume[volume_forecasts.Volume<0] = 0

sku_recommendations.SKU[0] = 'SKU_01'
sku_recommendations.SKU[1] = 'SKU_04'
sku_recommendations.SKU[2] = 'SKU_01'
sku_recommendations.SKU[3] = 'SKU_02'

volume_forecasts.to_csv('volume_forecast.csv', index= False)
sku_recommendations.to_csv('sku_recommendation.csv', index= False)



