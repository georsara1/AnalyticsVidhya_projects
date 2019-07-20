#Import modules
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data
train10 = pd.read_csv('train1.csv')
train9 = pd.read_csv('train9.csv')

test10 = pd.read_csv('test1.csv')
test9 = pd.read_csv('test9.csv')

hero_data = pd.read_csv('hero_data.csv')

submission_df = pd.read_csv('sample_submission.csv')

#Pre-process datasets
df_all = pd.concat([train9,train10, test9])
df_all = df_all.sort_values(by = 'user_id')
df_all = df_all.reset_index(drop = True)

train_final = pd.merge(df_all,hero_data, on = ['hero_id'], how='left')
test_final = pd.merge(test10,hero_data, on = ['hero_id'], how='left')

#Create a new variable counting the percentage of wins to games
train_final['wins2games'] = train_final.num_wins/ train_final.num_games

#Create a new dataframe that groups and summarizes player stats
player_stats1 = train_final.groupby(['user_id'], as_index = True)['wins2games'].mean()
player_stats2 = train_final.groupby(['user_id'], as_index = True)['num_games'].mean()
player_stats3 = train_final.groupby(['user_id'], as_index = True)['num_wins'].mean()

player_stats_all = pd.concat([player_stats1,
                              player_stats2,
                              player_stats3], axis =1)

#Append the stats to train set
train_final = train_final.set_index('user_id').join(player_stats_all, rsuffix = '_player_stats_all')
train_final['user_id'] = train_final.index
train_final = train_final.reset_index(drop = True)

#Do the same thing for the test set
test_final = test_final.set_index('user_id').join(player_stats_all, rsuffix = '_player_stats_all')
test_final['user_id'] = test_final.index
test_final = test_final.reset_index(drop = True)

#The test_final set does not include the wins2games information and the num_wins_player_stats_all information

#Merge datasets for pre-processing
train_y = train_final['kda_ratio'].astype('float')
train_final = train_final.drop(['num_wins','kda_ratio'], axis = 1)

train_test_set = pd.concat([train_final, test_final], axis = 0, ignore_index= True)

#--------------------------Feature engineering-------------------------------
#Count number of roles the hero has
train_test_set['No_of_roles'] = train_test_set.roles.str.count(':')+1

#Create a separate column for each role
train_test_set['Carry']=np.zeros([train_test_set.shape[0],1])
train_test_set['Nuker']=np.zeros([train_test_set.shape[0],1])
train_test_set['Disabler']=np.zeros([train_test_set.shape[0],1])
train_test_set['Escape']=np.zeros([train_test_set.shape[0],1])
train_test_set['Pusher']=np.zeros([train_test_set.shape[0],1])
train_test_set['Initiator']=np.zeros([train_test_set.shape[0],1])
train_test_set['Jungler']=np.zeros([train_test_set.shape[0],1])
train_test_set['Durable']=np.zeros([train_test_set.shape[0],1])
train_test_set['Support']=np.zeros([train_test_set.shape[0],1])

role_index = train_test_set['roles'].str.contains('Carry')
train_test_set.Carry[role_index] = 1
role_index = train_test_set['roles'].str.contains('Nuker')
train_test_set.Nuker[role_index] = 1
role_index = train_test_set['roles'].str.contains('Disabler')
train_test_set.Disabler[role_index] = 1
role_index = train_test_set['roles'].str.contains('Escape')
train_test_set.Escape[role_index] = 1
role_index = train_test_set['roles'].str.contains('Pusher')
train_test_set.Pusher[role_index] = 1
role_index = train_test_set['roles'].str.contains('Initiator')
train_test_set.Initiator[role_index] = 1
role_index = train_test_set['roles'].str.contains('Jungler')
train_test_set.Jungler[role_index] = 1
role_index = train_test_set['roles'].str.contains('Support')
train_test_set.Support[role_index] = 1
role_index = train_test_set['roles'].str.contains('Durable')
train_test_set.Durable[role_index] = 1

#Delete unndeeded variables
train_test_set = train_test_set.drop(['id','base_health', 'base_mana', 'base_mana_regen',
                                      'base_magic_resistance', 'roles', 'num_wins'], axis = 1)

#Set correct feature types
print('Assigning correct data types...')
train_test_set.user_id = train_test_set.user_id.astype('category')
train_test_set.hero_id = train_test_set.hero_id.astype('category')
#train_test_set.roles = train_test_set.roles.astype('category')
train_test_set.No_of_roles = train_test_set.No_of_roles.astype('category')
train_test_set.primary_attr = train_test_set.primary_attr.astype('category')
train_test_set.attack_type = train_test_set.attack_type.astype('category')

train_test_set.Carry = train_test_set.Carry.astype('category')
train_test_set.Nuker = train_test_set.Nuker.astype('category')
train_test_set.Disabler = train_test_set.Disabler.astype('category')
train_test_set.Escape = train_test_set.Escape.astype('category')
train_test_set.Pusher = train_test_set.Pusher.astype('category')
train_test_set.Initiator = train_test_set.Initiator.astype('category')
train_test_set.Jungler = train_test_set.Jungler.astype('category')
train_test_set.Durable = train_test_set.Durable.astype('category')
train_test_set.Support = train_test_set.Support.astype('category')

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = train_test_set.select_dtypes(include=['number']).copy()
var_non_numeric = train_test_set.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

train_test_set= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Split again in train and test sets
train_x = train_test_set.iloc[:29022,:]
test_x = train_test_set.iloc[29022:,:]

#Build LightGBM Model
train_data=lgb.Dataset(train_x,label=train_y)

param = {'num_leaves': 34, 'objective':'regression', 'max_depth':12,
         'learning_rate':.08,  'metric': 'rmse',
         'feature_fraction': 0.5}

cv_mod = lgb.cv(param,
                train_data,
                num_boost_round = 1000,
                #min_data = 1,
                nfold = 5,
                early_stopping_rounds = 20,
                verbose_eval=20,
                stratified = False,
                show_stdv=True,
                )

num_boost_rounds_lgb = len(cv_mod['rmse-mean'])

lgbm = lgb.train(param, train_data, num_boost_rounds_lgb)

ax = lgb.plot_importance(lgbm, max_num_features=37)
plt.show()

predictions = lgbm.predict(test_x)

submission_df.kda_ratio = predictions

lgbm_submission = submission_df.to_csv('lgbm_submission.csv', index = False)

#[260]	cv_agg's rmse:

#Leaderboard score:


