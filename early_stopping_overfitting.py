# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 09:23:13 2021

@author: akino

early stopping using one layer
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Import tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Let's see the first 3 images and corresponding labels in our training data
"""
images = X_train[:3]
labels = y_train[:3]

for index, image in enumerate(images):
    print ('Label:', labels[index])
    print ('Digit in the image', np.argmax(labels[index])) #argmax picks out the label with highest probability
    plt.imshow(image.reshape(28,28),cmap='gray')
    plt.show()
"""
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
##Tensorboard as we discussed is for visualisation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from datetime import datetime


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

###Sequential just symbolises
model1 = Sequential()
##input layer
model1.add(Dense(50, activation='relu', input_shape= (784,) ))

##we will use softmax as activation function because this is a multiclass classification problem.
### if you dont know what softmax , you need to understand it mathematically.
#Intuitively what it does is it converts numbers into probabilities.
###find out more here --> https://www.youtube.com/watch?v=p-XCC0y8eeY
model1.add(Dense(10, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

## for tensorboard visualisation you can check out the following link: https://youtu.be/Uzkhn5ENJzQ
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

training_history = model1.fit(
X_train, # input
y_train, # output
batch_size=32,
verbose=1, # Suppress chatty output; use Tensorboard instead
epochs=10,
validation_data=(X_test, y_test),
callbacks=[tensorboard_callback],
)

#adding early stopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model3 = Sequential()
model3.add(Dense(128, activation='relu', input_shape= (784,) ))
model3.add(Dense(256, activation='relu'))
model3.add(Dense(128, activation='relu'))
model3.add(Dense(10, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

training_history = model3.fit(
X_train, # input
y_train, # output
batch_size = 32,
verbose = 1, # Suppress chatty output; use Tensorboard instead
epochs = 10,
validation_data = (X_test, y_test),
callbacks = [tensorboard_callback, es],
)
 

