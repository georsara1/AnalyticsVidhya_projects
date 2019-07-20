import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import sqrt, floor


cv_scores =[]

for i in range(5):

    #Import data
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_submission = pd.read_csv('sample_submission.csv')

    df_test['Upvotes'] = 0

    df_train['is_train'] = 1
    df_test['is_train'] = 0

    #Concatenate into a single data frame
    df_all = pd.concat([df_train, df_test], axis = 0)

    #Check for null values
    df_all.isnull().sum().sum() #OK all imputed


    #Feature engineering and transformations
    df_all['Username'] = df_all['Username'].astype('object')

    for col in df_all.columns:
        if df_all[col].dtype == 'object':
            df_all[col] = df_all[col].astype('category')
            df_all[col] = df_all[col].cat.codes


    del df_all['ID']

    #df_all['answer_per_view'] = df_all['Answers'] / df_all['Views']
    df_all['views_per_rep'] = df_all['Views'] / df_all['Reputation']

    del df_all['Username']
    #del df_all['Views']

    df_all['Tag_Frequency'] = (df_all.groupby(['Tag'])['Tag'].transform('count'))/df_all.shape[0]
    #df_all['Tag_Number'] = (df_all.groupby(['Tag'])['Tag'].transform('count'))
    #del df_all['Tag']

    #Split in train, validation and test sets
    train = df_all[df_all['is_train'] == 1]
    test = df_all[df_all['is_train'] == 0]

    del train['is_train']
    del test['is_train']


    train_early_stop, valid_early_stop = train_test_split(train, test_size= 0.1)

    # train_early_stop = train.iloc[:300000,:]
    # valid_early_stop = train.iloc[300000:,:]

    train_early_stop_x = train_early_stop.drop(['Upvotes'], axis =1)
    train_early_stop_y = train_early_stop['Upvotes']

    valid_early_stop_x = valid_early_stop.drop(['Upvotes'], axis =1)
    valid_early_stop_y = valid_early_stop['Upvotes']


    del test['Upvotes']

    train_early_stop_x = train_early_stop_x.replace([np.inf, -np.inf], np.nan)
    valid_early_stop_x = valid_early_stop_x.replace([np.inf, -np.inf], np.nan)
    print(train_early_stop_x.isnull().sum().sum())

    train_early_stop_x['views_per_rep'] = train_early_stop_x['views_per_rep'].fillna(value = train_early_stop_x['views_per_rep'].median())
    valid_early_stop_x['views_per_rep'] = valid_early_stop_x['views_per_rep'].fillna(value = valid_early_stop_x['views_per_rep'].median())

    #-------------------Build model for validation-----------------
    regr = RandomForestRegressor(max_depth=8, bootstrap=False, max_leaf_nodes = 200)
    regr.fit(train_early_stop_x, train_early_stop_y)

    print(regr.feature_importances_)

    predictions = regr.predict(valid_early_stop_x)

    score=np.round(sqrt(mean_squared_error(valid_early_stop_y, predictions)),4)
    print("Final score on validation set: " + str(np.round(score,3)))
    cv_scores.append(score)

print('Average score:', np.mean(cv_scores))

#-------------------Build model for prediction-----------------

train_all_x = train.drop(['Upvotes'], axis =1)
train_all_y = train['Upvotes']

train_all_x = train_all_x.replace([np.inf, -np.inf], np.nan)
train_all_x['views_per_rep'] = train_all_x['views_per_rep'].fillna(value = train_all_x['views_per_rep'].median())


test_all_x = test.replace([np.inf, -np.inf], np.nan)
test_all_x['views_per_rep'] = test_all_x['views_per_rep'].fillna(value = test_all_x['views_per_rep'].median())

regr = RandomForestRegressor(max_depth=8, bootstrap=False,  max_leaf_nodes =4)
regr.fit(train_all_x, train_all_y)


predictions_final = regr.predict(test_all_x)

submission = pd.DataFrame({'ID': df_test['ID'], 'Upvotes': predictions_final})
submission = submission.sort_values(by = 'ID')

submission.to_csv('rf2.csv', index = False)

