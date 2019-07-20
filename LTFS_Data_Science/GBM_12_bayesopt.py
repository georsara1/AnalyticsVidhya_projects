# CV score: 0.67275   PL
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
from bayes_opt import BayesianOptimization

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


df_all['branch_employee'] = df_all['branch_id'].astype(str) + df_all['Employee_code_ID'].astype(str)
df_all['supplier_employee'] = df_all['supplier_id'].astype(str) + df_all['Employee_code_ID'].astype(str)
#df_all['supplier_branch'] = df_all['supplier_id'].astype(str) + df_all['branch_id'].astype(str)
#df_all['manufacturer_employee'] = df_all['manufacturer_id'].astype(str) + df_all['branch_id'].astype(str)

#------------Encoding categorical-----------
#OHE encoding
ohe_cols = ['State_ID',
            #'branch_id',
            'manufacturer_id',
            #'PERFORM_CNS.SCORE.DESCRIPTION',

            ]

df_all = pd.get_dummies(df_all, columns=ohe_cols)

label_encode_cols = ['branch_employee', 'supplier_employee']

for col in label_encode_cols:
    df_all[col] = df_all[col].astype('category')
    df_all[col] = df_all[col].cat.codes

cat_cols_count = ['Current_pincode_ID', 'Employee_code_ID',
                  #'branch_employee', 'supplier_employee',
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

count_encoded_vars = count_encode(df_all, cat_cols_count, normalize=True)
df_all = pd.concat([df_all, count_encoded_vars], axis = 1)

cat_cols_labelcount = ['Current_pincode_ID', 'Employee_code_ID',
            #'branch_employee', 'supplier_employee',
            ]

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

count_encoded_vars = labelcount_encode(df_all, cat_cols_labelcount)
df_all = pd.concat([df_all, count_encoded_vars], axis = 1)

cat_cols_freq = ['Current_pincode_ID', 'Employee_code_ID',
            'branch_employee', 'supplier_employee',
            ]

#Frequency encoding
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')


for col in tqdm(cat_cols_freq):
    df_all = frequency_encoding(df_all, col)


#------------Drop unneeded-----------------
df_all = df_all.drop(['MobileNo_Avl_Flag', 'Date.of.Birth', 'DisbursalDate', 'PERFORM_CNS.SCORE.DESCRIPTION',
                      'Passport_flag', 'Driving_flag',
                      'Employment.Type',
                      'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH'

                      ], axis = 1)

# #Clustering
# print('CLustering...')
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=4, random_state=0, init = 'random').fit_predict(df_all)
# df_all['kmeans'] = kmeans


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

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1989)
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

cat_cols_mean = ['Current_pincode_ID', 'Employee_code_ID',
            'branch_employee', 'supplier_employee',
            ]

for col in tqdm(cat_cols_mean):
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 7)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    train_df[new_feat_name] = temp_encoded_tr.values
    test_df[new_feat_name] = temp_encoded_te.values

#Drop the original categorical variables after encoding them
train_df = train_df.drop(['branch_employee', 'supplier_employee'], axis = 1)
test_df = test_df.drop(['branch_employee', 'supplier_employee'], axis = 1)

#PCA and clustering
cols_for_pca =['disbursed_amount', 'asset_cost', 'ltv',
       'supplier_id',  'Current_pincode_ID',
        'Employee_code_ID',
       'Aadhar_flag', 'PAN_flag', 'VoterID_flag',
       'PERFORM_CNS.SCORE',
       'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
       'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
       'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',
       'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
       'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',
       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
        'NO.OF_INQUIRIES']
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(train_df[cols_for_pca])
pca_comp_train = pca.transform(train_df[cols_for_pca]) #x_embedding=pca.transform(data) to visuallize
# fig,ax=plt.subplots(1,2,figsize=(10,4))
# pc_total=np.arange(1,pca.n_components_+1)
# ax[0].plot(pc_total,np.cumsum(pca.explained_variance_ratio_))
# ax[0].set_xticks(pc_total)
# ax[0].set_xlabel('Principal Components')
# ax[0].set_ylabel('Cumulative explained variance')
# ###############################################################
# ax[1].plot(pc_total,pca.explained_variance_)
# ax[1].set_xticks(pc_total)
# ax[1].set_xlabel('Principal Components')
# ax[1].set_ylabel('Explained Variance Ratio')
# fig.suptitle('A GENERAL SCREE PLOT')
# plt.show()
pca_comp_test = pca.transform(test_df[cols_for_pca])

train_df['pca1'] = pca_comp_train[:,0]
train_df['pca2'] = pca_comp_train[:,1]
#train_df['pca3'] = pca_comp_train[:,2]

test_df['pca1'] = pca_comp_test[:,0]
test_df['pca2'] = pca_comp_test[:,1]
#test_df['pca3'] = pca_comp_test[:,2]

#-----------Modelling part---------
param = {
    #'bagging_freq': 5,
    #'bagging_fraction': 0.635,
    #'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.3592708878130908,
    'learning_rate': 0.06061266744826507,
    'max_depth': -1,
    'metric':'auc',
    #'min_data_in_leaf': 80,
    #'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 17,
    'reg_alpha': 6.944935638610119,
    'reg_lambda': 3.0910767281631935,
    #'num_threads': 8,
    #'tree_learner': 'serial',
    'objective': 'binary',
    #'is_unbalance': True,
    'verbosity': -1,
}

features = [c for c in train_df.columns if c not in ['UniqueID', 'loan_default']]
target = train_df['loan_default']

X = train_df[features]
y = train_df['loan_default']
train_data = lgb.Dataset(X, label=y)
def lgb_eval(num_leaves, lambda_l1, lambda_l2, feature_fraction, learning_rate, min_data_in_leaf):
    params = {'application':'binary','num_iterations':4000, 'early_stopping_round':30, 'metric':'auc'}
    params["num_leaves"] = int(round(num_leaves))
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['min_data_in_leaf'] = int(round(min_data_in_leaf))
    #params['max_depth'] = int(round(max_depth))
    params['learning_rate'] = max(learning_rate,0)
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    #params['min_split_gain'] = min_split_gain
    #params['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(params, train_data, nfold=4, seed=1422, stratified=True, verbose_eval =200, metrics=['auc'])
    return max(cv_result['auc-mean'])

lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (10, 18),
                                        'feature_fraction': (0.3, 0.7),
                                        'min_data_in_leaf': (70, 90),
                                        #'max_depth': (5, 8.99),
                                        'learning_rate': (0.06,0.1),
                                        'lambda_l1': (3, 7),
                                        'lambda_l2': (3, 7),
                                        #'min_split_gain': (0.001, 0.1),
                                        #'min_child_weight': (5, 50)
                                        }, random_state=0)

lgbBO.maximize(init_points=15, n_iter=10)

#Find set of best parameters
l = [lgbBO.res[i]['target'] for i in range(len(lgbBO.res))]
max_idx = [i for i,j in enumerate(l) if j == np.max(l)][0]
opt_param = lgbBO.res[max_idx]['params']

#Correct opt params where needed and update dictionary with basic parameters
opt_param['num_leaves'] = int(opt_param['num_leaves'])
opt_param['min_data_in_leaf'] = int(opt_param['min_data_in_leaf'])
params = {'application':'binary','num_iterations':4000, 'early_stopping_round':30, 'metric':'auc'}
opt_param.update(params)

#Train model
nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=4950)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(opt_param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                    early_stopping_rounds=30, categorical_feature=['branch_id'])
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / nfold

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

# #Print importances
# cols = (feature_importance_df[["feature", "importance"]]
#         .groupby("feature")
#         .mean()
#         .sort_values(by="importance", ascending=False)[:1000].index)
# best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
# plt.figure(figsize=(10,10))
# sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
# plt.title('LightGBM Features (averaged over folds)')
# plt.tight_layout()

#Predict and write to submit
predictions_01 = np.where(predictions>0.5,1,0)

submission['loan_default'] = predictions

submission.to_csv('submission_lgb12.csv', index = False)