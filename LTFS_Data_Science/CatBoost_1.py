 #AUC: 0.6715196078336483 PL 0.659064268933837
# AUC: 0.67260 PL 0.65951

from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import time
from datetime import timedelta, date
from tqdm import tqdm

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


#-------Set correct data types------------


#----------Feature engineering---------
df_all['ltv2'] = df_all['disbursed_amount']/df_all['asset_cost']
df_all['ltv_ratio'] = df_all['ltv2']/df_all['ltv']



#------------Encoding categorical-----------
#OHE encoding
ohe_cols = ['State_ID',
            'branch_id',
            'manufacturer_id',
            #'PERFORM_CNS.SCORE.DESCRIPTION',

            ]

df_all = pd.get_dummies(df_all, columns=ohe_cols)

#label_encode_cols = ['PERFORM_CNS.SCORE.DESCRIPTION']

# for col in label_encode_cols:
#     df_all[col] = df_all[col].astype('category')
#     df_all[col] = df_all[col].cat.codes

cat_cols = ['Current_pincode_ID', 'Employee_code_ID',
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

#Frequency encoding
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')


for col in tqdm(cat_cols):
    df_all = frequency_encoding(df_all, col)


#------------Drop unneeded-----------------
df_all = df_all.drop(['MobileNo_Avl_Flag', 'Date.of.Birth', 'DisbursalDate', 'PERFORM_CNS.SCORE.DESCRIPTION',
                      'Passport_flag', 'Driving_flag',
                      'Employment.Type',
                      'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH',

                      ], axis = 1)

#-----Split again in train and test-------
train_df = df_all[df_all['is_train']==1]
test_df = df_all[df_all['is_train']==0]

del train_df['is_train']
del test_df['is_train']


def mean_k_fold_encoding(col, alpha):
    target_name = 'loan_default'
    target_mean_global = train_df[target_name].mean()

    nrows_cat = train_df.groupby(col)[target_name].count()
    target_means_cats = train_df.groupby(col)[target_name].mean()
    target_means_cats_adj = (target_means_cats * nrows_cat +
                             target_mean_global * alpha) / (nrows_cat + alpha)
    # Mapping means to test data
    encoded_col_test = test_df[col].map(target_means_cats_adj)

    kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=1989)
    parts = []
    for trn_inx, val_idx in kfold.split(train_df,train_df['loan_default']):
        df_for_estimation, df_estimated = train_df.iloc[trn_inx], train_df.iloc[val_idx]
        nrows_cat = df_for_estimation.groupby(col)[target_name].count()
        target_means_cats = df_for_estimation.groupby(col)[target_name].mean()

        target_means_cats_adj = (target_means_cats * nrows_cat +
                                 target_mean_global * alpha) / (nrows_cat + alpha)

        encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
        parts.append(encoded_col_train_part)

    encoded_col_train = pd.concat(parts, axis=0)
    encoded_col_train.fillna(target_mean_global, inplace=True)
    encoded_col_train.sort_index(inplace=True)

    return encoded_col_train, encoded_col_test


for col in tqdm(cat_cols):
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    train_df[new_feat_name] = temp_encoded_tr.values
    test_df[new_feat_name] = temp_encoded_te.values


#Split in 75% train and 25% test set
features = [c for c in train_df.columns if c not in [

                    'UniqueID', 'loan_default']]

train_early_x, valid_early_x, train_early_y, valid_early_y = train_test_split(train_df[features],
                                                                              train_df['loan_default'],
                                                                              test_size = 0.25,
                                                                              random_state= 1984)

print("\nCatBoost...")
cb_model = CatBoostClassifier(iterations=1000,
                              learning_rate=0.083,
                              #depth=7,
                              l2_leaf_reg=5,
                              bootstrap_type='Bernoulli',
                              subsample=0.4,
                              #scale_pos_weight=4,
                              eval_metric='AUC',
                              metric_period=50,
                              od_type='Iter',
                              od_wait=45,
                              random_seed=17,
                              allow_writing_files=False)

cb_model.fit(train_early_x, train_early_y,
             eval_set=(valid_early_x, valid_early_y),
             #cat_features=ohe_cols,
             use_best_model=True,
             #verbose=True
             )

# fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': cols})
# fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
# _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
# plt.savefig('catboost_feature_importance.png')

print('AUC:', roc_auc_score(valid_early_y, cb_model.predict_proba(valid_early_x)[:, 1]))
y_preds = cb_model.predict_proba(test_df[features])[:, 1]

submission['loan_default'] = y_preds

submission.to_csv('cat_boost_1.csv', index = False)