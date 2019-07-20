'''
Epoch 30/30
loss: 0.1987 - acc: 0.9275 - val_loss: 0.5050 - val_acc: 0.8829
PL 0.83
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications import VGG16, VGG19, InceptionV3, InceptionResNetV2, ResNet50, MobileNet, MobileNetV2
from keras import regularizers

batch_size = 16
image_size = 200

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        #vertical_flip=True
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (image_size, image_size, 3))
conv_base.trainable = False

model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(conv_base)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Activation('softmax'))

for layer in conv_base.layers:
    #print(layer)
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

for layer in conv_base.layers[-3:]:
    #print(layer)
    layer.trainable = True

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=70,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training


# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# summarize history for loss
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


#Import test images
path = 'data/test'
image_set_test = []
for img in os.listdir(path):
    #character_name = simpson
    #next_label = [label for label, character in char_dict.items() if character == character_name][0]
    next_img = cv2.imread(path + '/' + img)
    image_set_test.append(cv2.resize(next_img, (image_size, image_size)))

image_set_test = np.array(image_set_test)
image_set_test=image_set_test/255.0

preds = model.predict(image_set_test)
preds_argmax = np.argmax(preds,axis=1)
preds_argmax2 = preds_argmax+1

submission = pd.read_csv('sample_submission.csv')
submission['category'] = preds_argmax2
submission.to_csv('keras_guide_tl_3.csv', index = False)
