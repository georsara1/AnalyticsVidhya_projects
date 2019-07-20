
#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
sns.set_style("whitegrid")
import statsmodels.api as sm

#Import data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

#Check missing
train_df.isnull().sum()
test_df.isnull().sum()

#Data info
train_df.describe(include = 'all')
train_df.gender.unique()
train_df.Residence_type.unique()
train_df.hypertension.unique()
train_df.heart_disease.unique()
train_df.ever_married.unique()
train_df.work_type.unique()
train_df.Residence_type.unique()


#---------------------------Vizualizations-----------------------------
#Distribution of variable 'age' versus stroke
plt.figure()
fig1 = sns.violinplot(y = train_df.age, x = train_df.stroke)
plt.show(fig1)

#scatter of variable 'Age' versus bmi
plt.figure()
fig2 = sns.regplot(x="age", y="bmi", data=train_df)
plt.show(fig2)

#scatter of variable 'Age' versus glucose
plt.figure()
fig2 = sns.regplot(x="age", y="avg_glucose_level", data=train_df)
plt.show(fig2)

#scatter of variable 'bmi' versus glucose
plt.figure()
fig2 = sns.regplot(x="bmi", y="avg_glucose_level", data=train_df)
plt.show(fig2)

#---------------------------Pre-processing-------------------------
print('Pre-processing full data set...')
# Concatenate data sets
test_df['stroke'] = 0
train_df['is_train'] = 1
test_df['is_train'] = 0

df_all = pd.concat([train_df, test_df], axis = 0)
df_all = df_all.reset_index(drop = True)

#In null of smoking status if work_type is children replace with never smoked
df_all.smoking_status[df_all.work_type == 'children'] = 'never smoked'

#In gender replace other with woman since its more common
# df_all.gender[df_all.gender == 'Other'] = 'Female'

#Create a linear regression model to impute bmi missing values based on age

# Define the predictor variable as separate dataframe
# reg_df = pd.DataFrame(df_all.age[df_all.bmi.notnull()], columns=['age'])
# # Put the target in another DataFrame
# reg_target = pd.DataFrame(df_all.bmi[df_all.bmi.notnull()], columns=["bmi"])
#
# X = reg_df["age"]
# y = reg_target["bmi"]
#
# # Create OLS model
# model = sm.OLS(y, X).fit()
# # Print out the statistics
# model.summary()

#Use the equation of the regression to impute bmi missing values
# df_all['bmi']= df_all.apply(
#     lambda row:
#             0.5584 * row.age if np.isnan(row.bmi) else row.bmi, axis=1)

#Set correct data types
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_columns:
    df_all[col] = df_all[col].astype('category')

for col in categorical_columns:
    df_all[col] = df_all[col].cat.codes


df_all = df_all.drop(['id'], axis = 1)

#Split in train and test set
train = df_all[df_all.is_train == 1]
test = df_all[df_all.is_train == 0]

train = train.drop(['is_train'], axis = 1)
test = test.drop(['is_train', 'stroke'], axis = 1)

#--------------------Create validation for early stopping of Light GBM train procedure------------
train_early_stop, valid_early_stop = train_test_split(train, test_size= 0.1, random_state= 7)

#-------------------------------------------------------------------------------------------------
train_early_stop_y = train_early_stop.stroke
train_early_stop_x = train_early_stop.drop(['stroke'], axis = 1)

valid_early_stop_y = valid_early_stop.stroke
valid_early_stop_x = valid_early_stop.drop(['stroke'], axis = 1)


test_x = test

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_early_stop_x,label=train_early_stop_y)
valid_data=lgb.Dataset(valid_early_stop_x,label=valid_early_stop_y)

pos = len(train_df.stroke[train_df.stroke == 0]) / len(train_df.stroke[train_df.stroke == 1])

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 94,
          'num_class' : 1,
          'metric' : 'auc'
          }

# Create parameters to search
gridParams = {
    'learning_rate': [0.05],
    #'n_estimators': [8,16],
    'max_depth': [6, 8, 10],
    'num_leaves': [16, 20, 24],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501],
    'colsample_bytree' : [0.64, 0.65],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [ 1.2],
    'reg_lambda' : [ 1.4],
    'scale_pos_weight': [58,76,94]
    }

# Create classifier to use. Note that parameters have to be input manually, not as a dict!
mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 5,
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

# To view the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=1, cv=9, n_jobs=-1)

# Run the grid
grid.fit(train_early_stop_x, train_early_stop_y)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
params['max_depth'] = grid.best_params_['max_depth']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']
# params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']
params['scale_pos_weight'] = grid.best_params_['scale_pos_weight']

print('Fitting with params: ')
print(params)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 early_stopping_rounds= 40,
                 valid_sets=valid_data,
                 verbose_eval= 4
                 )
print('best_number of iterations:', lgbm.best_iteration)

#Predict on test set
predictions_lgbm_prob = lgbm.predict(test_x,  num_iteration=lgbm.best_iteration)
predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
percentage_of_ones = np.sum(predictions_lgbm_01)/len(predictions_lgbm_01)
print(percentage_of_ones)

#--------------------------Print accuracy measures and variable importances----------------------
#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')

submission_df.stroke = predictions_lgbm_prob
submission_df.to_csv('lgb_submission.csv', index = False)