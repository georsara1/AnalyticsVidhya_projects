from nltk.tokenize import word_tokenize
import itertools
import pandas as pd
import numpy as np
import re
import string
import collections

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Flatten, Dropout, Convolution1D
from keras.layers.embeddings import Embedding

#Import data and split into train and testdatasets
print('Importing data set...')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sample_submission.csv')

#Import regex to strip punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))

#Initiate Lemmatizer
word_lemm = WordNetLemmatizer()

#---------------------------Tokenize sentences--------------------------
#1. Train set
text_list_train = list(df_train['tweet'])
text_list_train_regex = [regex.sub('', sentence) for sentence in text_list_train]
tokenized_text_train = [word_tokenize(i) for i in text_list_train_regex]
tokenized_text_train = [[re.sub('[^A-Za-z]','', word) for word in sentence] for sentence in tokenized_text_train]
tokenized_text_train = [[word_lemm.lemmatize(word) for word in sentence] for sentence in tokenized_text_train]

#2. Test set
text_list_test = list(df_test['tweet'])
text_list_test_regex = [regex.sub('', sentence) for sentence in text_list_test]
tokenized_text_test = [word_tokenize(i) for i in text_list_test_regex]
tokenized_text_test = [[re.sub('[^A-Za-z]','', word) for word in sentence] for sentence in tokenized_text_test]
tokenized_text_test = [[word_lemm.lemmatize(word) for word in sentence] for sentence in tokenized_text_test]

#Create vocabulary from train set only
list_of_all_words = list(itertools.chain.from_iterable(tokenized_text_train))
vocabulary =sorted(list(set(list_of_all_words)))

#Remove most common words
counter=collections.Counter(list_of_all_words)
#print(counter.most_common(20))

#Remove stopwords and rest disturbances
stop = set(stopwords.words('english'))
vocabulary = [word for word in vocabulary if word not in stop]
vocabulary.remove('user')
vocabulary = vocabulary[40:]

#Create a dictionary from the vocabulary
vocabulary_dict = dict(zip(np.array(vocabulary), range(len(vocabulary)+1)))

#--------------------------Pre-processing train and test sets----------------------------------
max_words_in_sentence=18 #max is 34

#------------TRAIN SET---------------
print('Building tokenized train  database...')
tokdf_train = pd.DataFrame(tokenized_text_train)

if tokdf_train.shape[1]>max_words_in_sentence:
    tokdf_train = tokdf_train.drop(tokdf_train.columns[[range(max_words_in_sentence,tokdf_train.shape[1])]], axis=1)
else:
    for col in range(tokdf_train.shape[1],max_words_in_sentence):
        tokdf_train[col]=0

for col in tokdf_train.columns:
    tokdf_train[col] = tokdf_train[col].map(vocabulary_dict)

tokdf_train.fillna(value = 0, inplace= True)

#--------------TEST SET-------------
print('Building tokenized test database...')
tokdf_test = pd.DataFrame(tokenized_text_test)

if tokdf_test.shape[1]>max_words_in_sentence:
    tokdf_test = tokdf_test.drop(tokdf_test.columns[[range(max_words_in_sentence,tokdf_test.shape[1])]], axis=1)
else:
    for col in range(tokdf_test.shape[1],max_words_in_sentence):
        tokdf_test[col]=0


for col in tokdf_test.columns:
    tokdf_test[col] = tokdf_test[col].map(vocabulary_dict)

tokdf_test.fillna(value = 0, inplace= True)

#------------------------------End of Pre-processing----------------------------------------------------
validation_results = []

for i in range(5):
    #Define train and Test sets
    train_x, valid_x, train_y, valid_y = train_test_split(tokdf_train, df_train['label'], test_size = 0.15, random_state= 4)

    train_x = np.array(train_x)
    valid_x = np.array(valid_x)
    test_x = np.array(tokdf_test)

    l=len(vocabulary)+1
    inp=train_x.shape[1]

    #Build a Convolutional Network model
    model = Sequential()
    model.add(Embedding(l, 96, input_length=inp))
    model.add(Convolution1D(96, 2, padding='valid'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Fit the model
    model.fit(train_x, train_y, validation_split= 0.2, epochs=3, batch_size=64, verbose=2)

    # Check performance on validation set
    valid_sentiments = model.predict(valid_x)
    valid_sentiments[valid_sentiments<0.5]=0
    valid_sentiments[valid_sentiments>0.5]=1
    valid_sentiments = valid_sentiments.astype(int)

    f1_score_cv = f1_score(valid_y, valid_sentiments)
    print('Validation F1 score', i+1, ':', f1_score_cv)
    validation_results.append(f1_score_cv)

print('Mean Validation F1 score:', np.mean(validation_results))

#---------------Build model with the entire train set to submit-------------------
#Define train and Test sets
train_x = np.array(tokdf_train)
train_y = np.array(df_train['label'])

#Build a Convolutional Network model
model = Sequential()
model.add(Embedding(l, 64, input_length=inp))
model.add(Convolution1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_x, train_y, validation_split= 0.2, epochs=3, batch_size=64, verbose=2)

# Predict
test_sentiments = model.predict(test_x)
test_sentiments[test_sentiments<0.5]=0
test_sentiments[test_sentiments>0.5]=1
test_sentiments = test_sentiments.astype(int)

df_submission['id'] = df_test['id']
df_submission['label'] = test_sentiments

df_submission.to_csv('cnn2.csv', index = False)

