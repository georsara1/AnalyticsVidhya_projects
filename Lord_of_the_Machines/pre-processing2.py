
#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from datetime import datetime
sns.set_style("whitegrid")
np.random.seed(697)

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
campaign_data_df = campaign_data_df.drop(['email_body', 'subject', 'email_url'], axis = 1)

#--------------------------- Pre-process train and test sets--------------------
print('Pre-processing full data set...')
# Concatenate data sets
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['is_click'] = 0

#train_df = train_df.drop(['is_open'], axis = 1)
df_all = pd.concat([train_df, test_df], axis = 0)
df_all = df_all.reset_index(drop = True)

# Split send_date into Month, day_of_month, day_of_week and time
#df_all['month'] = df_all.send_date.str[3:5]
#df_all['day_of_month'] = df_all.send_date.str[:2]
#df_all['time'] = df_all.send_date.str[11:13]

# dt2 = []
# for i in range(df_all.shape[0]):
#     dt2.append(datetime.strptime(df_all.send_date[i], "%d-%m-%Y %H:%M").weekday())
# df_all['day_of_week'] = dt2

# Get some info about opening the mail according to campaign
campaind_id_only_in_train = campaign_data_df[campaign_data_df['campaign_id'].isin(train_df.campaign_id.unique())]
campaind_id_only_in_train = pd.merge(campaign_data_df, train_df[['campaign_id','is_open', 'user_id']],on='campaign_id', how='right')
is_open_by_user_id = campaind_id_only_in_train.groupby(by = ['user_id'])['is_open'].mean()
is_open_by_user_id = is_open_by_user_id.to_frame(name = 'is_open_by_user_id')
is_open_by_user_id['user_id'] = is_open_by_user_id.index
df_all_final = pd.merge(df_all,is_open_by_user_id[['user_id','is_open_by_user_id']],on='user_id', how='left')

# Drop unneeded variables
df_all_final = df_all_final.drop(['id', 'is_open', 'send_date'], axis = 1)

#Encode categorical variables to ONE-HOT
# print('Converting categorical variables to numeric...')
# categorical_columns = [ 'campaign_id'
#                        #, 'recipient_type', 'email_url', 'subject_mood', 'month', 'day_of_month','is_user_new', 'time',
#                        ]
#
# df_all_final = pd.get_dummies(df_all_final, columns = categorical_columns,
#                     #drop_first = True #Does not affect the algorithm's performance
#                     )

#Set correct types to the respectful variables
df_all_final['campaign_id'] = df_all_final['campaign_id'].astype('category')
df_all_final['user_id'] = df_all_final['user_id'].astype('category')

#Scale continuous variables to [0,1] range
# print('Normalizing...')
# columns_to_scale = ['total_links', 'no_of_internal_links', 'no_of_images',
#                     'day_of_week', 'day_of_month', 'user_id',
#                     'no_of_sections', 'email_body_lengths', 'email_subject_lengths'
#                     #,'is_open_by_communication_type'
#                     ]
#
# df_all_final[columns_to_scale]=df_all_final[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))

# Save to csv for future use
#df_all_final.to_csv('df_all_final.csv')
df_all_final.to_pickle('df_all_final_pickled')
