
#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb


#Import dataset
train_df = pd.read_csv('Train.csv', sep = ',')
test_df = pd.read_csv('Test.csv', sep = ',')
submission_df = pd.read_csv('SampleSubmission.csv', sep = ',')

train_y = train_df['Item_Outlet_Sales'].values
train_df = train_df.drop(['Item_Outlet_Sales'], axis = 1)

#Exploratory Data Analysis
print('Doing some pre-processing to the data...')
train_test_set = pd.concat([train_df, test_df], axis = 0, ignore_index= True)

train_test_set['Item_Fat_Content'] = train_test_set['Item_Fat_Content'].astype('category')
train_test_set['Item_Fat_Content'][train_test_set['Item_Fat_Content']=='LF'] = 'Low Fat'
train_test_set['Item_Fat_Content'][train_test_set['Item_Fat_Content']=='low fat'] = 'Low Fat'
train_test_set['Item_Fat_Content'][train_test_set['Item_Fat_Content']=='reg'] = 'Regular'

train_test_set['Item_Type'] = train_test_set['Item_Type'].astype('category')

train_test_set['Outlet_Establishment_Year'] = train_test_set['Outlet_Establishment_Year'].astype('category')

train_test_set['Outlet_Identifier'] = train_test_set['Outlet_Identifier'].astype('category')

train_test_set['Outlet_Location_Type'] = train_test_set['Outlet_Location_Type'].astype('category')

train_test_set['Outlet_Type'] = train_test_set['Outlet_Type'].astype('category')

train_test_set['Outlet_Size'] = train_test_set['Outlet_Size'].astype('category')

train_test_set['Item_Identifier'] = train_test_set['Item_Identifier'].astype('category')

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = train_test_set.select_dtypes(include=['number']).copy()
var_non_numeric = train_test_set.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

train_test_set= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Split again in train and test set
train_set = train_test_set.iloc[:8523].copy()
test_set = train_test_set.iloc[8523:].copy()

#Build LightGBM Model
train_data=lgb.Dataset(train_set,label=train_y)

param = {'num_leaves': 14,
         'objective':'regression',
         'max_depth':4,
         'learning_rate':.08, 'metric': 'rmse',
         'feature_fraction': 0.8,
         # 'zero_as_missing': False
         }

cv_mod = lgb.cv(param,
                train_data,
                num_boost_round = 1000,
                #min_data = 1,
                nfold = 5,
                early_stopping_rounds = 20,
                verbose_eval=100,
                #stratified = True,
                #show_stdv=True,
                )


num_boost_rounds_lgb = len(cv_mod['multi_error-mean'])

lgbm = lgb.train(param, train_data, 88)

ax = lgb.plot_importance(lgbm, max_num_features=21)
plt.show()

p = lgbm.predict(test_set)

submission_df['Item_Outlet_Sales'] = p

#submission_df.to_csv('my_submission.csv', sep = ',',index = False)

