
#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from datetime import datetime
sns.set_style("whitegrid")
np.random.seed(697)
from sklearn.model_selection import GridSearchCV
from datetime import datetime
sns.set_style("whitegrid")
np.random.seed(697)
import matplotlib.pyplot as plt
import xgboost as xgb


#Import data
print('Importing data...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
campaign_data_df = pd.read_csv('campaign_data.csv')
submission_df = pd.read_csv('sample_submission.csv')

#---------------------------Pre-processing campaign data-------------------------
print('Pre-processing campaign data...')
# Get length of each e-mail body
e_mail_body_words = list(campaign_data_df.email_body)
e_mail_body_words = [word_tokenize(i) for i in e_mail_body_words]
e_mail_body_lengths = [len(w) for w in e_mail_body_words]
campaign_data_df['email_body_lengths'] = e_mail_body_lengths

# Get length of each e-mail Subject
e_mail_subject_words = list(campaign_data_df.subject)
e_mail_subject_words = [word_tokenize(i) for i in e_mail_subject_words]
e_mail_subject_lengths = [len(w) for w in e_mail_subject_words]
campaign_data_df['email_subject_lengths'] = e_mail_subject_lengths

# Drop unneeded variables
campaign_data_df = campaign_data_df.drop(['email_body', 'subject', 'email_url',
                                          'no_of_images',
                                          #'total_links',
                                          'no_of_internal_links',
                                          #'communication_type',
                                          #'email_body_lengths',
                                          #'email_subject_lengths'
                                          ], axis = 1)

#--------------------------- Pre-process train and test sets--------------------
print('Pre-processing full data set...')
# Concatenate data sets
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['is_click'] = 0

#Keep in train set only the user ids that are contained in the test set
#train_df = train_df[train_df.user_id.isin(test_df['user_id'])]

#train_df = train_df.drop(['is_open'], axis = 1)
df_all = pd.concat([train_df, test_df], axis = 0)
df_all = df_all.reset_index(drop = True)

# Split send_date into Month, day_of_month, day_of_week and time
#df_all['month'] = df_all.send_date.str[3:5]
#df_all['day_of_month'] = df_all.send_date.str[:2]
df_all['time'] = df_all.send_date.str[11:13]

dt2 = []
for i in range(df_all.shape[0]):
    dt2.append(datetime.strptime(df_all.send_date[i], "%d-%m-%Y %H:%M").weekday())
df_all['day_of_week'] = dt2

# Get some info about opening the mail according to campaign
campaind_id_only_in_train = campaign_data_df[campaign_data_df['campaign_id'].isin(train_df.campaign_id.unique())]
campaind_id_only_in_train = pd.merge(campaign_data_df, train_df[['campaign_id','is_open', 'user_id']],on='campaign_id', how='right')
is_open_by_user_id = campaind_id_only_in_train.groupby(by = ['user_id'])['is_open'].mean() #mean works best
is_open_by_user_id = is_open_by_user_id.to_frame(name = 'is_open_by_user_id')
is_open_by_user_id['user_id'] = is_open_by_user_id.index
df_all_final = pd.merge(df_all, is_open_by_user_id[['user_id','is_open_by_user_id']],on='user_id', how='left', sort=False)

#Import campaign_id characteristics
df_all_final = pd.merge(df_all_final,campaign_data_df, on = ['campaign_id'], how='left')

# Drop unneeded variables
df_all_final = df_all_final.drop(['id', 'user_id','campaign_id', 'is_open', 'send_date', 'no_of_sections'], axis = 1)

#Set correct types to the respectful variables
#df_all_final['user_id'] = df_all_final['user_id'].astype('category')
#df_all_final['user_id'] = df_all_final['user_id'].cat.codes
df_all_final['communication_type'] = df_all_final['communication_type'].astype('category')
df_all_final['communication_type'] = df_all_final['communication_type'].cat.codes
#df_all_final['day_of_month'] = df_all_final['day_of_month'].astype('int')
df_all_final['time'] = df_all_final['time'].astype('int')

#Scale continuous variables to [0,1] range
print('Normalizing...')
columns_to_scale = ['is_open_by_user_id',
                    'total_links',
                    #'day_of_month',
                    'email_body_lengths',
                    'email_subject_lengths',
                    #'no_of_sections', 'email_body_lengths', 'email_subject_lengths',
                    #'communication_type'
                    ]

df_all_final[columns_to_scale]=df_all_final[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))


#Split in train and test set
train_x = df_all_final[df_all_final.is_train == 1]
test_x = df_all_final[df_all_final.is_train == 0]

train_y = train_x.is_click

train_x = train_x.drop(['is_train', 'is_click'], axis=1)
test_x = test_x.drop(['is_train', 'is_click'], axis=1)

#------------------------Build LightGBM Model-----------------------
dtrain=xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

scale_perc = (train_df.shape[0]-train_df.is_click.sum())/train_df.is_click.sum()

params = {'learning_rate': 0.1,
          'tree_method': "auto",
          'grow_policy': "lossguide",
          'num_leaves': 700,
          'max_depth': 1,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'min_child_weight': 0,
          #'alpha':4,
          'objective': 'binary:logistic',
          'scale_pos_weight': scale_perc,
          'eval_metric': 'auc',
          'nthread':8,
          'random_state': 99,
          'silent': False}

xgb_model = xgb.train(params, dtrain, 30, maximize=True, verbose_eval=2)

ax = xgb.plot_importance(xgb_model, max_num_features=21)
plt.show()

#Make predictions
predictions_xgb_prob = xgb_model.predict(dtest)
plt.hist(predictions_xgb_prob)
predictions_lgbm_01 = np.where(predictions_xgb_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
predictions_lgbm_01.sum()/len(predictions_lgbm_01)

#Write to csv to predict
submission_df.is_click = predictions_xgb_prob
submission_df.to_csv('xgb_submission.csv', index = False)