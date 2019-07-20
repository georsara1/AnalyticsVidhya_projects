
#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from nltk.tokenize import word_tokenize
from keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
sns.set_style("whitegrid")
np.random.seed(697)
from datetime import datetime

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
                                          #'no_of_images',
                                          #'total_links',
                                          #'no_of_internal_links',
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

#train_df = train_df.drop(['is_open'], axis = 1)
df_all = pd.concat([train_df, test_df], axis = 0)
df_all = df_all.reset_index(drop = True)

# Split send_date into Month, day_of_month, day_of_week and time
# df_all['month'] = df_all.send_date.str[3:5]
# df_all['month'][df_all['month']== '01'] = '13'
# df_all['month'][df_all['month']== '02'] = '14'
# df_all['month'][df_all['month']== '03'] = '15'
#df_all['day_of_month'] = df_all.send_date.str[:2]
#df_all['time'] = df_all.send_date.str[11:13]

# dt2 = []
# for i in range(df_all.shape[0]):
#     dt2.append(datetime.strptime(df_all.send_date[i], "%d-%m-%Y %H:%M").weekday())
# df_all['day_of_week'] = dt2

# Get some info about opening the mail according to campaign
campaind_id_only_in_train = campaign_data_df[campaign_data_df['campaign_id'].isin(train_df.campaign_id.unique())]
campaind_id_only_in_train = pd.merge(campaign_data_df, train_df[['campaign_id','is_open', 'user_id']],on='campaign_id', how='right')
is_open_by_user_id = campaind_id_only_in_train.groupby(by = ['user_id'])['is_open'].mean() #mean works best
is_open_by_user_id = is_open_by_user_id.to_frame(name = 'is_open_by_user_id')
is_open_by_user_id['user_id'] = is_open_by_user_id.index
df_all_final = pd.merge(df_all, is_open_by_user_id[['user_id','is_open_by_user_id']],on='user_id', how='left')

#Import campaign_id characteristics
df_all_final = pd.merge(df_all_final,campaign_data_df, on = ['campaign_id'], how='left')

# Drop unneeded variables
df_all_final = df_all_final.drop(['id', 'user_id','campaign_id', 'is_open','no_of_sections','no_of_images',
                                  'send_date', 'communication_type','no_of_internal_links',
                                  ], axis = 1)

#Set correct types to the respectful variables
#df_all_final['user_id'] = df_all_final['user_id'].astype('category')
#df_all_final['communication_type'] = df_all_final['communication_type'].astype('category')
#df_all_final['day_of_month'] = df_all_final['day_of_month'].astype('int')
#df_all_final['month'] = df_all_final['month'].astype('int')
#df_all_final['time'] = df_all_final['time'].astype('int')

#Encode categorical variables to ONE-HOT
# print('Converting categorical variables to numeric...')
# categorical_columns = [ 'communication_type'
#                        #, 'recipient_type', 'email_url', 'subject_mood', 'month', 'day_of_month','is_user_new', 'time',
#                        ]
#
# df_all_final = pd.get_dummies(df_all_final, columns = categorical_columns,
#                     #drop_first = True #Does not affect the algorithm's performance
#                     )

#Scale continuous variables to [0,1] range
print('Normalizing...')
columns_to_scale = ['is_open_by_user_id', 'total_links','email_body_lengths', 'email_subject_lengths'

                     #'email_body_lengths', 'email_subject_lengths','time','day_of_week',
                    #,'is_open_by_communication_type'
                    ]

df_all_final[columns_to_scale]=df_all_final[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))


#Split in train and test set
train_x = df_all_final[df_all_final.is_train == 1]
test_x = df_all_final[df_all_final.is_train == 0]

train_y = train_x.is_click

train_x = train_x.drop(['is_train', 'is_click'], axis=1)
test_x = test_x.drop(['is_train', 'is_click'], axis=1)

train_x = np.array(train_x)
test_x = np.array(test_x)

#-------------------Build the Neural Network model-------------------
print('Building Neural Network model...')
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adam = optimizers.adam(lr = 0.005, decay = 0.0000001)

model = Sequential()
model.add(Dense(4, input_dim=train_x.shape[1],
                kernel_initializer='normal',
                kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
# model.add(Dense(32,
#                 #kernel_regularizer=regularizers.l2(0.02),
#                 activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='Nadam')

history = model.fit(train_x, train_y, validation_split=0.2, epochs=5, batch_size=32)

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Predict on test set
predictions_NN_prob = model.predict(test_x)
#predictions_NN_prob = predictions_NN_prob[:,0]
plt.figure()
plt.hist(predictions_NN_prob)
plt.show()
predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
predictions_NN_01.sum()

#Write to csv to predict
submission_df.is_click = predictions_NN_prob
submission_df.to_csv('NN_submission.csv', index = False)



