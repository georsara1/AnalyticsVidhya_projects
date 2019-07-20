#CV score: 0.83551 LB 0.8323
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing
from keras import backend as K
from sklearn.preprocessing import StandardScaler

#Import data
import gc
print('Importing data..')
train_df = pd.read_csv('train.csv', nrows = 1000000)
test_df = pd.read_csv('test.csv')
user_features_df = pd.read_csv('user_features.csv')
submission = pd.read_csv('sample_submission_only_headers.csv')

print('Merging train set...')
train_df = train_df.merge(user_features_df, left_on='node1_id', right_on='node_id')
train_df = train_df.merge(user_features_df, left_on='node2_id', right_on='node_id')

print('Merging test set...')
test_df = test_df.merge(user_features_df, left_on='node1_id', right_on='node_id')
test_df = test_df.merge(user_features_df, left_on='node2_id', right_on='node_id')

#Empty memory
del user_features_df
gc.collect()

test_df = test_df.sort_values(by = 'id')

#Drop unneeded features
ids = test_df['id']
y_label = train_df['is_chat']
train_df = train_df.drop(['node1_id', 'node2_id',  'node_id_x','node_id_y', 'is_chat'], axis = 1)
test_df = test_df.drop(['id', 'node1_id', 'node2_id', 'node_id_x','node_id_y'], axis = 1)

#Split in 75% train and 25% test set
train_x, test_x, train_y, test_y = train_test_split(train_df, y_label, test_size = 0.25, random_state= 1984)

sc = StandardScaler()
sc.fit(train_x)
train_x = sc.transform(train_x)
test_x = sc.transform(test_x)

#-------------------Build the Neural Network model-------------------
print('Building Neural Network model...')
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.adam(lr = 0.008,
                       #decay = 0.0000001
                            )

#Custom function
def get_weighted_loss():
    def weighted_loss(y_true, y_pred):
        return K.mean( K.square(y_pred - y_true) * K.exp(-K.log(1.7) * (K.log(1. + K.exp((y_true - 3)/5 )))),axis=-1  )
    return weighted_loss

model = Sequential()
model.add(Dense(26, input_dim=train_x.shape[1],
                #kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="tanh"))
model.add(Dropout(0.1))
model.add(Dense(26,
                #kernel_regularizer=regularizers.l2(0.02),
                activation="tanh"))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation("sigmoid"))
#model.compile(loss=get_weighted_loss(), optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam')

history = model.fit(train_x, train_y, validation_split=0.2, epochs=30, batch_size=128, verbose = 2)

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
predictions_NN_prob = predictions_NN_prob[:,0]

predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

#Print accuracy
acc_NN = accuracy_score(test_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)

#Print Area Under Curve
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

#Print Confusion Matrix
cm = confusion_matrix(test_y, predictions_NN_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
