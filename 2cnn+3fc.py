# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:42:02 2018

@author: admin
"""

import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split

#import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) =fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore and preprocessing the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
#reshape the input data 
train_images = train_images.reshape(60000, 28, 28, -1)
test_images = test_images.reshape(10000, 28, 28, -1)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                    test_size=0.15, 
                                                    random_state=0)


# Build the model
model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(BatchNormalization())

model.add(Conv2D(64, (5, 5), activation='relu', padding='same', 
                 bias_initializer='RandomNormal', 
                 kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
              metrics=['accuracy'])


import time 
start_time = time.time()

hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
               batch_size=256, epochs=30, verbose=1)
training_time = time.time() - start_time

print(hist.history)
model.evaluate(test_images, test_labels)
model.save('my_2cnn3fc_model.h5')

#training time
mm = training_time // 60
ss = training_time % 60
print('Training {} epochs in {}:{}'.format(10, int(mm), round(ss, 1)))


#plot loss and accuracy
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#loss plot
tra = plt.plot(loss)
val = plt.plot(val_loss, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(['Training', 'Validation'], loc=1)
plt.show()

# accuracy plot
plt.plot(acc)
plt.plot(val_acc, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Accuracy')
plt.legend(['Training', 'Validation'], loc=4)
plt.show()


#增加BN层，测试精度为0.8974
model2 = Sequential()
model2.add(InputLayer(input_shape=(28, 28, 1)))
model2.add(Conv2D(64, (5, 5), activation='relu', padding='same', 
                 bias_initializer='RandomNormal', 
                 kernel_initializer='random_uniform'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))

optimizer = Adam(lr=0.001)
model2.compile(loss='categorical_crossentropy', optimizer=optimizer, 
              metrics=['accuracy'])


import time 
start_time = time.time()

hist = model2.fit(x_train, y_train, validation_data=(x_val, y_val), 
               batch_size=256, epochs=30, verbose=1)
training_time = time.time() - start_time




