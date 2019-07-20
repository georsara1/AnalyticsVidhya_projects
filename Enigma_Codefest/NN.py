import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers, regularizers
from math import sqrt, floor
from keras.optimizers import adam,adagrad,rmsprop

#Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sample_submission.csv')

df_test['Upvotes'] = 0

df_train['is_train'] = 1
df_test['is_train'] = 0

#Concatenate into a single data frame
df_all = pd.concat([df_train, df_test], axis = 0)


#-----------------------Feature engineering-------------------
df_all['answer_per_view'] = df_all['Answers'] / df_all['Views']
df_all['views_per_rep'] = df_all['Views'] / df_all['Reputation']
#df_all['Tag_Frequency'] = (df_all.groupby(['Tag'])['Tag'].transform('count'))/df_all.shape[0]
#df_all['User_count'] = df_all.groupby(['Username'])['Username'].transform('count')

#Label or OHE encoding
# df_all['Username'] = df_all['Username'].astype('category')
# df_all['Username'] = df_all['Username'].cat.codes

#df_all = pd.get_dummies(df_all, columns = ['Tag'])

# df_all['Username'] = df_all['Username'].astype('object')
# for col in df_all.columns:
#     if df_all[col].dtype == 'object':
#         df_all[col] = df_all[col].astype('category')
#         df_all[col] = df_all[col].cat.codes


del df_all['ID']
del df_all['Username']
#del df_all['Views']
del df_all['Tag']


#-----------------Split in train, validation and test sets-------------
train = df_all[df_all['is_train'] == 1]
test = df_all[df_all['is_train'] == 0]

del train['is_train']
del test['is_train']

train_early_stop_x, valid_early_stop_x = train_test_split(train, test_size= 0.15, random_state= 78)

train_early_stop_y = train_early_stop_x['Upvotes']
del train_early_stop_x['Upvotes']

valid_early_stop_y = valid_early_stop_x['Upvotes']
del valid_early_stop_x['Upvotes']

del test['Upvotes']

train_early_stop_x = train_early_stop_x.replace([np.inf, -np.inf], np.nan)
print(train_early_stop_x.isnull().sum().sum())
train_early_stop_x['views_per_rep'] = train_early_stop_x['views_per_rep'].fillna(value = train_early_stop_x['views_per_rep'].median())


col = ['Reputation', 'Answers', 'Views', 'answer_per_view', 'views_per_rep']
train_early_stop_x[col]=train_early_stop_x[col].apply(lambda x: (x-x.min())/(x.max()-x.min()))
train_early_stop_x = np.array(train_early_stop_x)

#-----------------------------Build model-----------------------------
adam_opt = adam(lr = 0.01)

model = Sequential()
model.add(Dense(48, input_dim=train_early_stop_x.shape[1],
                kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.1),
                activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, input_dim=train_early_stop_x.shape[1],
                kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.2),
                activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(8,
                kernel_regularizer=regularizers.l2(0.2),
                kernel_initializer='uniform',
                activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=adam_opt)

history = model.fit(train_early_stop_x, train_early_stop_y, validation_split=0.2, epochs=60, batch_size=32, verbose = 2)

rmse_train_loss = [sqrt(loss) for loss in history.history['loss']]
rmse_val_loss = [sqrt(loss) for loss in history.history['val_loss']]

# summarize history for loss
plt.figure()
plt.plot(rmse_train_loss)
plt.plot(rmse_val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

predictions = model.predict(valid_early_stop_x)

score=np.round(sqrt(mean_squared_error(valid_early_stop_y, predictions)),4)
print("Final score on validation set: " + str(np.round(score,3)))


#---------------------------Predict on test set-----------------------
train_x = train.drop(['Upvotes'], axis = 1)
train_y = train['Upvotes']

model = Sequential()
model.add(Dense(90, input_dim=train_x.shape[1],
                kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
# model.add(Dropout(0.1))
model.add(Dense(16,
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
model.add(Dropout(0.3))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='adam')

history = model.fit(train_x, train_y, validation_split=0.2, epochs=15, batch_size=64, verbose = 2)

predictions = model.predict(test)
predictions = [floor(p) if p>=0 else 0 for p in predictions]

#valid_rmse = sqrt(mean_squared_error(valid_early_stop_y, predictions))

submission = pd.DataFrame({'ID': df_test['ID'], 'Upvotes': predictions})
submission = submission.sort_values(by = 'ID')

df_submission['Upvotes'] = predictions

submission.to_csv('LightGBM_simple_sorted_2.csv', index = False)



