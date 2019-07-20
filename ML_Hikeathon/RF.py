
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import roc_curve, auc,accuracy_score,confusion_matrix,make_scorer

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

# sc = StandardScaler()
# sc.fit(train_x)
# train_x = sc.transform(train_x)
# test_x = sc.transform(test_x)

all_cols = ['f1_x', 'f2_x', 'f3_x', 'f4_x', 'f5_x', 'f6_x', 'f7_x', 'f8_x', 'f9_x',
       'f10_x', 'f11_x', 'f12_x', 'f13_x', 'f1_y', 'f2_y', 'f3_y', 'f4_y',
       'f5_y', 'f6_y', 'f7_y', 'f8_y', 'f9_y', 'f10_y', 'f11_y', 'f12_y',
       'f13_y']

features = [ 'f13_x', 'f13_y', 'f3_x', 'f12_x', 'f12_y', 'f3_y', 'f1_y', 'f11_x', 'f4_x', 'f1_x']

clf = ExtraTreesClassifier(n_estimators=250, max_depth=18, criterion= 'entropy', max_features = 0.5)
clf.fit(train_x[features],train_y)

prediction = clf.predict_proba(test_x[features])
prediction = prediction[:,1]

print('Validation Accuracy: ')
fpr,tpr,_ = roc_curve(test_y, prediction)
print(auc(fpr,tpr))