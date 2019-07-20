#Best num_boost_round:
#Best CV score:
#LB:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

#Import data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

#Create separate dataframe for Y label
y_label = train_df['renewal']
del train_df['renewal']

#Concatenate into one dataframe
data = pd.concat([train_df,test_df])

#Drop variables we cannot use
data = data.drop(['id'], axis = 1)

#Feature Engineering
#data['age_in_days'][data['age_in_days']>95*365] = data['age_in_days'][data['age_in_days']<95*365].median()
#data['Income'][data['Income']>250000] = data['Income'][data['Income']<250000].median()

#One hot (or label) encoding
categorical_features = [col for col in data.columns if data[col].dtype == 'object']
#one_hot_data = pd.get_dummies(data, columns=categorical_features)
for col in categorical_features:
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes

#Split in train and test set
train_x = data.iloc[:train_df.shape[0],:]
test_x = data.iloc[train_df.shape[0]:,:]
train_y = y_label.iloc[:train_df.shape[0]]
#test_y = y_label.iloc[train_df.shape[0]:]

#Create train and validation set
#train_lgb_x, valid_lgb_x, train_lgb_y, valid_lgb_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=True)



#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
#valid_data=lgb.Dataset(valid_lgb_x,label=valid_lgb_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          #'max_depth' : 4,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 16,
          'learning_rate': 0.05,
          'max_bin': 512,
          #'subsample_for_bin': 200,
          #'subsample': 1,
          'subsample_freq': 5,
          'colsample_bytree': 0.8,
          'reg_alpha': 1.2,
          'reg_lambda': 1.4,
          'min_split_gain': 0.6,
          #'min_child_weight': 0.8,
          #'min_child_samples': 5,
          #'scale_pos_weight': 1,
          'is_unbalance': True,
          'num_class' : 1,
          'metric' : 'auc'
          }

cv_mod = lgb.cv(params,
                train_data,
                500,
                nfold = 10,
                early_stopping_rounds = 25,
                stratified = True)

print('Current parameters:\n', params)
print('\nBest num_boost_round:', len(cv_mod['auc-mean']))
print('Best CV score:', cv_mod['auc-mean'][-1])

num_iter = len(cv_mod['auc-mean'])

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 num_iter,
                 #valid_sets=valid_data,
                 #early_stopping_rounds= 40,
                 verbose_eval= 10
                 )


#Predict on test set and write to submit
predictions_lgbm_prob = lgbm.predict(test_x)
pred_01 = np.where(predictions_lgbm_prob>0.5,1,0)


#Plot Variable Importances
#lgb.plot_importance(lgbm, max_num_features=21, importance_type='gain')

submission['renewal'] = predictions_lgbm_prob

#Strategy 1
# submission['incentives'] = 400
# submission['incentives'][submission['renewal']<0.4] = 0

#Strategy 2
# submission['incentives'] = 400
# submission['incentives'][submission['renewal']<0.4] = 0
# submission['incentives'][submission['renewal']>0.95] = 0

#Strategy 3
# submission['incentives'] = 400
# submission['incentives'][submission['renewal']<0.4] = 0
# submission['incentives'][submission['renewal']>0.9] = 350
# submission['incentives'][submission['renewal']>0.95] = 300

#Strategy 3_2
# submission['incentives'] = 500
# submission['incentives'][submission['renewal']<0.4] = 0
# submission['incentives'][submission['renewal']>0.9] = 400
# submission['incentives'][submission['renewal']>0.95] = 350

#Strategy 4
#Define the functions as instructed
def effort_calc(incentive):
    effort = 10*(1-np.exp(-incentive/400))
    return effort

def improve_calc(effort):
    improve_ren = 20*(1-np.exp(-effort/5))/100
    return improve_ren

def total_eq(p, improve_dp, premium, incentive):
    y = (p + p*improve_dp)*premium-incentive
    return y

#Calculate incentives for each ocassion
test_premium = test_x['premium']
test_p = predictions_lgbm_prob

final_incentive_list = []
final_dp_list =[]

for f in range(len(test_p)):
    incentive_list = []
    dp_list = []
    y_list = []
    if f%1000 == 0:
        print('We are at row', f)
    for i in range(0, 300, 10):
        effort = effort_calc(i)
        improve_dp = improve_calc(effort)

        y = total_eq(test_p[f], improve_dp, test_premium[f], i)

        if improve_dp < (1 - test_p[f]):
            incentive_list.append(i)
            dp_list.append(improve_dp)
            y_list.append(y)

    incentive_to_agent = incentive_list[np.argmax(y_list)]
    dp = dp_list[np.argmax(y_list)]

    final_incentive_list.append(incentive_to_agent)
    final_dp_list.append(dp)

submission['incentives'] = final_incentive_list

#Check initial versus final net revenue
net_revenue_initial = test_x['premium']*pred_01
print('Initial net revenue:', sum(net_revenue_initial))

final_dp_list_sum = predictions_lgbm_prob + np.array(final_dp_list)
final_dp_list_01 = np.where(final_dp_list_sum>0.5,1,0)

net_revenue_new = test_x['premium']*final_dp_list_01 - submission['incentives']
print('Net revenue after incentives:', sum(net_revenue_new))


submission.to_csv('Submission_Lgbm3.csv', index = False)