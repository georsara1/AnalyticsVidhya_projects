
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

#Import data
import gc
print('Importing data..')
train_df = pd.read_csv('train.csv', nrows=100000)
test_df = pd.read_csv('test.csv')
user_features_df = pd.read_csv('user_features.csv')
submission = pd.read_csv('sample_submission_only_headers.csv')


#User features wrangling
#user_features_df['f1_4_7'] = user_features_df['f1'] + user_features_df['f4'] + user_features_df['f7']
#user_features_df['f5_8'] =  user_features_df['f5'] + user_features_df['f8']
#user_features_df['f2_5'] = (user_features_df['f2'] + user_features_df['f5'])
#user_features_df['f3_8'] = (user_features_df['f3'] + user_features_df['f8'])/2
#user_features_df['f6_8'] = (user_features_df['f6'] + user_features_df['f8'])/2
#user_features_df['f5_6'] = (user_features_df['f5'] + user_features_df['f6'])/2
#user_features_df['f9_12'] = (user_features_df['f9'] + user_features_df['f12'])/2

user_features_df = user_features_df.drop([#'f1',
                                          #'f2',
                                          #'f3',
                                          #'f4',
                                          #'f5',
                                          #'f6',
                                          #'f7',
                                          #'f8',
                                          #'f9',
                                          #'f12'
                                          ], axis = 1)

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
train_df = train_df.drop(['node1_id', 'node2_id',  'node_id_x','node_id_y'], axis = 1)
test_df = test_df.drop(['node1_id', 'node2_id', 'node_id_x','node_id_y'], axis = 1)

# #Stats
# train_df['sum'] = train_df.iloc[:,:26].sum(axis = 1)
# train_df['mean'] = train_df.iloc[:,:26].mean(axis = 1)
# train_df['std'] = train_df.iloc[:,:26].std(axis = 1)
# train_df['var'] = train_df.iloc[:,:26].var(axis = 1)
# #number of zeros>>???
# test_df['sum'] = test_df.iloc[:,:26].sum(axis = 1)
# test_df['mean'] = test_df.iloc[:,:26].mean(axis = 1)
# test_df['std'] = test_df.iloc[:,:26].std(axis = 1)
# test_df['var'] = test_df.iloc[:,:26].var(axis = 1)

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.5,
    'learning_rate': 0.03,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'lambda_l2':5,
    'lambda_l1':5,
    #'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1,
    #'is_unbalance': True
}

features = [c for c in train_df.columns if c not in [   # 'f2_y',
                                                        # 'f3_x',
                                                        # 'f4_y',
                                                        # 'f6_y',
                                                        'ID_code', 'is_chat']]

target = y_label
nfold = 5

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=89)
oof = train_df[['is_chat']]
oof['predict'] = 0
predictions = test_df[['id']]
val_aucs = []
feature_importance_df = pd.DataFrame()

X_test = test_df[features].values

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['is_chat'])):
    X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx]['is_chat']
    X_valid, y_valid = train_df.iloc[val_idx][features], train_df.iloc[val_idx]['is_chat']

    N = 5
    p_valid, yp = 0, 0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')

        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(param,
                            trn_data,
                            100000,
                            valid_sets=[trn_data, val_data],
                            early_stopping_rounds=200,
                            verbose_eval=100,
                            evals_result=evals_result
                            )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid / N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)

    predictions['fold{}'.format(fold + 1)] = yp / N

mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

# sub = pd.DataFrame({'id':ids,
#                    'is_chat':predictions})
# sub.reset_index(drop = True, inplace=True)
# sub['is_chat'] = np.round(sub['is_chat'],2)
# sub['is_chat'] = sub['is_chat'].astype('float16')
# sub.to_csv('gbm4.csv', index = False)