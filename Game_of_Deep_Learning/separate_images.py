import pandas as pd
import os
import cv2
from shutil import copyfile, copy2

path = "images"

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sample_submission.csv')

#Create lists according to category
images_1 = df_train['image'][df_train['category']==1].values
images_2 = df_train['image'][df_train['category']==2].values
images_3 = df_train['image'][df_train['category']==3].values
images_4 = df_train['image'][df_train['category']==4].values
images_5 = df_train['image'][df_train['category']==5].values


for img in os.listdir(path):
    if img in images_1:
        copy2('images/' + img, 'images_1')
    elif img in images_2:
        copy2('images/' + img, 'images_2')
    elif img in images_3:
        copy2('images/' + img, 'images_3')
    elif img in images_4:
        copy2('images/' + img, 'images_4')
    elif img in images_5:
        copy2('images/' + img, 'images_5')


#Create test set
images_test = df_test['image'].values

for img in os.listdir(path):
    if img in images_test:
        copy2('images/' + img, 'test')