# CV score: 0.66575  PL 0.6599212948
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

#Import data
print('Importing data..')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

#Concatenate data sets for preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0

test_df['loan_default'] = 0

df_all = pd.concat([train_df, test_df], axis= 0)


#----------Feature engineering---------
df_all['ltv2'] = df_all['disbursed_amount']/df_all['asset_cost']
df_all['ltv_diff'] = df_all['ltv2']/df_all['ltv']

#------------Encoding categorical-----------
#OHE encoding
ohe_cols = ['State_ID',
            'branch_id',
            'manufacturer_id'
            ]

df_all = pd.get_dummies(df_all, columns=ohe_cols)


cat_cols = ['Current_pincode_ID', 'Employee_code_ID'
            ]

#Count encoding
def count_encode(X, categorical_features, normalize=False):
    print('Count encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        X_[cat_feature] = X[cat_feature].astype(
            'object').map(X[cat_feature].value_counts())
        if normalize:
            X_[cat_feature] = X_[cat_feature] / np.max(X_[cat_feature])
    X_ = X_.add_suffix('_count_encoded')
    if normalize:
        X_ = X_.astype(np.float32)
        X_ = X_.add_suffix('_normalized')
    else:
        X_ = X_.astype(np.uint32)
    return X_

count_encoded_vars = count_encode(df_all, cat_cols, normalize=True)
df_all = pd.concat([df_all, count_encoded_vars], axis = 1)

#LabelCount Encodings
def labelcount_encode(X, categorical_features, ascending=False):
    print('LabelCount encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = X[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # for ascending ordering
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            # for descending ordering
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        X_[cat_feature] = X[cat_feature].map(
            labelcount_dict)
    X_ = X_.add_suffix('_labelcount_encoded')
    if ascending:
        X_ = X_.add_suffix('_ascending')
    else:
        X_ = X_.add_suffix('_descending')
    X_ = X_.astype(np.uint32)
    return X_

count_encoded_vars = labelcount_encode(df_all, cat_cols)
df_all = pd.concat([df_all, count_encoded_vars], axis = 1)


#------------Drop unneeded-----------------
df_all = df_all.drop(['MobileNo_Avl_Flag', 'Date.of.Birth', 'DisbursalDate', 'PERFORM_CNS.SCORE.DESCRIPTION',
                      'Passport_flag', 'Driving_flag', 'Employment.Type',
                      'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH',
                      #'Current_pincode_ID'
                      ], axis = 1)

#-----Split again in train and test-------
train_df = df_all[df_all['is_train']==1]
test_df = df_all[df_all['is_train']==0]

del train_df['is_train']
del test_df['is_train']



#-----------Modelling part---------
features = [c for c in train_df.columns if c not in [
                    'UniqueID', 'loan_default']]

target = train_df['loan_default']

h2o.init()
train_df = h2o.H2OFrame(train_df)

train, valid, test = train_df.split_frame(ratios=[0.6,0.2], seed=1234)
response = "loan_default"
train[response] = train[response].asfactor()
valid[response] = valid[response].asfactor()
test[response] = test[response].asfactor()
print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])

gbm = H2OGradientBoostingEstimator()
gbm.train(x=features, y=response, training_frame=train)
print(gbm)

perf = gbm.model_performance(valid)
print(perf)

gbm_tune = H2OGradientBoostingEstimator(
    ntrees = 3000,
    learn_rate = 0.01,
    stopping_rounds = 20,
    stopping_metric = "AUC",
    col_sample_rate = 0.7,
    sample_rate = 0.7,
    seed = 1234
)
gbm_tune.train(x=features, y=response, training_frame=train, validation_frame=valid)

pred = gbm.predict(test)
pred[:]

from h2o.automl import H2OAutoML

aml = H2OAutoML(max_models = 10, max_runtime_secs=300, seed = 1)
aml.train(x=features, y=response, training_frame=train, validation_frame=valid)