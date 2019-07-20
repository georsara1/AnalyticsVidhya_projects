import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import lightgbm as lgb
import nltk

print('Importing data...')
df_train = pd.read_csv('train.csv', sep = ',')
df_test = pd.read_csv('test.csv', sep = ',')
submission_df = pd.read_csv('sample_submission.csv', sep = ',')


#Merge into one data set for data wrangling
df_train['is_train'] = 1
df_test['is_train'] = 0

train_test_set = pd.concat([df_train, df_test], axis=0, ignore_index=True)

tokens = nltk.word_tokenize('go')
pos = nltk.pos_tag(tokens)
pos[0][1]

train_test_set['pos_tag'] = 'O'
for i in range(train_test_set.shape[0]):
    tokens = nltk.word_tokenize(train_test_set['Word'].loc[i])
    pos = nltk.pos_tag(tokens)
    train_test_set['pos_tag'].loc[0] = pos[0][1]