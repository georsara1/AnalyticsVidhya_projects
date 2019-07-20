
#---------------------Import Libraries------------------------------------
print('Importing needed libraries...')
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import imread
from scipy.misc import imresize
import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Flatten, Dropout, Convolution2D, MaxPooling2D, regularizers
from keras.layers.embeddings import Embedding

#------------------Import train and test sets------------------------------
print('Importing data sets...')
data_dir =  os.path.abspath('C:/Users/geosara1/Desktop/Python/Indian_Actors_Age')

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

i = random.choice(train.index)

img_name = train.ID[i]
img = imread(os.path.join(data_dir, 'Train', img_name))

print('Age: ', train.Class[i])
plt.imshow(img)
plt.show()

#-----------------------Resize all images to same size----------------------
#Train set
print('Resizing all images to same size (Train)...')
temp = []
for img_name in train.ID:
    img_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(img_path)
    img = imresize(img, (48, 48))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

train_x = np.stack(temp)

#Test set
print('Resizing all images to same size (Test)...')
temp = []
for img_name in test.ID:
    img_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(img_path)
    img = imresize(img, (48, 48))
    temp.append(img.astype('float32'))


#----------------------Create train and test sets--------------------
test_x = np.stack(temp)

train_x = train_x / 255.
test_x = test_x / 255.


lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)

'''
input_num_units = (48, 48, 3)
hidden_num_units = 360
output_num_units = 3

epochs = 3
batch_size = 8

#Build a simple FFNN
model = Sequential([
  InputLayer(input_shape=input_num_units),
  Flatten(),
  Dense(units=hidden_num_units, activation='relu'),
  Dense(units=output_num_units, activation='softmax'),
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, validation_split=0.22, batch_size=batch_size,epochs=epochs,verbose=2)
'''

#---------------------Build a Convolutional Network model-------------------------
print('Building the best model....')
model = Sequential()
#model.add(Embedding(48, (3,3), input_length=3))
model.add(Convolution2D(48, (3,3), input_shape=(48, 48, 3), padding='same',activation='relu'))
#model.add(Convolution2D(48, (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Convolution2D(96, (3,3), padding='same',activation='relu'))
#model.add(Convolution2D(96, (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Convolution2D(48, (3,3), padding='same',activation='relu'))
#model.add(Convolution2D(48, (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
print('Fitting the best model....')
model.fit(train_x, train_y, validation_split=0.22, epochs=5, batch_size=16, verbose=2)

#Make prediction and write to file
pred = model.predict_classes(test_x)
pred = lb.inverse_transform(pred)

test['Class'] = pred
test.to_csv('sub02.csv', index=False)