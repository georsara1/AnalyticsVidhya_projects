#Import modules
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 1.5)

#Import data
train10 = pd.read_csv('train1.csv')
train9 = pd.read_csv('train9.csv')

test10 = pd.read_csv('test1.csv')
test9 = pd.read_csv('test9.csv')

hero_data = pd.read_csv('hero_data.csv')

submission_df = pd.read_csv('sample_submission.csv')

train9_y = train9.kda_ratio
train10_y = train10.kda_ratio
test9_y = test9.kda_ratio

#--------------------------Feature engineering for hero stats table-----------------------------
#Count number of roles the hero has
hero_data['No_of_roles'] = hero_data.roles.str.count(':')+1

#Create a separate column for each role
hero_data['Carry']=np.zeros([hero_data.shape[0],1])
hero_data['Nuker']=np.zeros([hero_data.shape[0],1])
hero_data['Disabler']=np.zeros([hero_data.shape[0],1])
hero_data['Escape']=np.zeros([hero_data.shape[0],1])
hero_data['Pusher']=np.zeros([hero_data.shape[0],1])
hero_data['Initiator']=np.zeros([hero_data.shape[0],1])
hero_data['Jungler']=np.zeros([hero_data.shape[0],1])
hero_data['Durable']=np.zeros([hero_data.shape[0],1])
hero_data['Support']=np.zeros([hero_data.shape[0],1])

role_index = hero_data['roles'].str.contains('Carry')
hero_data.Carry[role_index] = 1
role_index = hero_data['roles'].str.contains('Nuker')
hero_data.Nuker[role_index] = 1
role_index = hero_data['roles'].str.contains('Disabler')
hero_data.Disabler[role_index] = 1
role_index = hero_data['roles'].str.contains('Escape')
hero_data.Escape[role_index] = 1
role_index = hero_data['roles'].str.contains('Pusher')
hero_data.Pusher[role_index] = 1
role_index = hero_data['roles'].str.contains('Initiator')
hero_data.Initiator[role_index] = 1
role_index = hero_data['roles'].str.contains('Jungler')
hero_data.Jungler[role_index] = 1
role_index = hero_data['roles'].str.contains('Support')
hero_data.Support[role_index] = 1
role_index = hero_data['roles'].str.contains('Durable')
hero_data.Durable[role_index] = 1

hero_data = hero_data.drop(['base_health', 'base_mana','base_mana_regen',
                            'base_magic_resistance', 'roles'], axis = 1)

hero_data.Carry = hero_data.Carry.astype('category')
hero_data.Nuker = hero_data.Nuker.astype('category')
hero_data.Disabler = hero_data.Disabler.astype('category')
hero_data.Escape = hero_data.Escape.astype('category')
hero_data.Pusher = hero_data.Pusher.astype('category')
hero_data.Initiator = hero_data.Initiator.astype('category')
hero_data.Jungler = hero_data.Jungler.astype('category')
hero_data.Durable = hero_data.Durable.astype('category')
hero_data.Support = hero_data.Support.astype('category')
#hero_data.hero_id = hero_data.hero_id.astype('category')
hero_data.primary_attr = hero_data.primary_attr.astype('category')
hero_data.attack_type = hero_data.attack_type.astype('category')

#Create a new variable counting the percentage of wins to games
train9['wins2games'] = train9.num_wins/ train9.num_games
train10['wins2games'] = train10.num_wins/ train10.num_games
test9['wins2games'] = test9.num_wins/ test9.num_games

#Test10 does not have the wins2games feature. We should add it
player_stats_test9 = test9.groupby(['user_id'], as_index = True)['wins2games'].mean()
test10 = test10.set_index('user_id').join(player_stats_test9, rsuffix = '_player_stats_all')
test10['user_id'] = test10.index

#Test10 does not have the wins feature. We should add it
player_stats_test9 = test9.groupby(['user_id'], as_index = True)['num_wins'].mean()
test10 = test10.set_index('user_id').join(player_stats_test9, rsuffix = '_player_stats_all')
test10['user_id'] = test10.index

#Delete Variable 'ID' and the target 'KDA_ratio' from all tables
train9 = train9.drop(['id',
                      #'num_wins',
                      'kda_ratio'], axis = 1)
train10 = train10.drop(['id',
                        #'num_wins',
                        'kda_ratio'], axis = 1)
test9 = test9.drop(['id',
                    #'num_wins',
                     'kda_ratio'], axis = 1)
test10 = test10.drop(['id'], axis = 1)

#Append the hero stats to all tables
train9 = pd.merge(train9,hero_data, on = ['hero_id'], how='left')
train10 = pd.merge(train10,hero_data, on = ['hero_id'], how='left')
test9 = pd.merge(test9,hero_data, on = ['hero_id'], how='left')
test10 = pd.merge(test10,hero_data, on = ['hero_id'], how='left')

#Merge datasets for pre-processing
train_all = pd.concat([train9, train10], axis = 0, ignore_index= True)
train_all_y = np.concatenate((train9_y,train10_y), axis = 0)

#Set correct feature types in train_all set
print('Assigning correct data types...')
train_all.user_id = train_all.user_id.astype('category')
train_all.hero_id = train_all.hero_id.astype('category')
train_all.No_of_roles = train_all.No_of_roles.astype('category')
train_all.primary_attr = train_all.primary_attr.astype('category')
train_all.attack_type = train_all.attack_type.astype('category')

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = train_all.select_dtypes(include=['number']).copy()
var_non_numeric = train_all.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

train_all= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Set correct feature types in test9 set
print('Assigning correct data types...')
test9.user_id = test9.user_id.astype('category')
test9.hero_id = test9.hero_id.astype('category')
test9.No_of_roles = test9.No_of_roles.astype('category')
test9.primary_attr = test9.primary_attr.astype('category')
test9.attack_type = test9.attack_type.astype('category')

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = test9.select_dtypes(include=['number']).copy()
var_non_numeric = test9.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

test9= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Set correct feature types in test10 set
print('Assigning correct data types...')
test10.user_id = test10.user_id.astype('category')
test10.hero_id = test10.hero_id.astype('category')
test10.No_of_roles = test10.No_of_roles.astype('category')
test10.primary_attr = test10.primary_attr.astype('category')
test10.attack_type = test10.attack_type.astype('category')

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = test10.select_dtypes(include=['number']).copy()
var_non_numeric = test10.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

test10= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Build XGBoost Model
train_final = pd.concat([train_all, test9], axis = 0, ignore_index= True)
train_final_y = np.concatenate((train9_y,train10_y, test9_y), axis = 0)
#train_final_y = pd.Series(train_final_y)

train_final=np.array(train_final)
test10=np.array(test10)

xgdmat = xgb.DMatrix(train_final, train_final_y)
testdmat = xgb.DMatrix(test10)

gxb_param = {'eta': 0.08, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'reg:linear', 'max_depth':8, 'min_child_weight':1}

cv_xgb = xgb.cv(params = gxb_param,
                dtrain = xgdmat,
                num_boost_round = 3000,
                nfold = 5,
                stratified= True,
                verbose_eval = 1,
                metrics = ['rmse'],
                show_stdv =True,
                early_stopping_rounds = 20)

final_gb = xgb.train(gxb_param, xgdmat, num_boost_round = 629)

xgb.plot_importance(final_gb)
importances = final_gb.get_fscore()
importances

y_pred = final_gb.predict(testdmat)

submission_df.kda_ratio = y_pred

submission_df.to_csv('XGBoost_submission.csv', index = False)

#[260]	cv_agg's rmse:

#Leaderboard score:


