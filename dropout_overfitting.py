# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 10:25:23 2021

@author: akino
"""

from tensorflow.keras.datasets import mnist
# import matplotlib for visualization
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

images = X_train[:3]
labels = y_train[:3]

f,ax = plt.subplots(nrows=1,ncols=3, figsize=(20,4))

for index,(img, ax) in enumerate(zip(images, ax)):

    ax.imshow(img,cmap='gray')
    ax.axis('off')
    ax.text(0.6,-2.0, f"Digit in the Image {labels[index]}", size=15, ha="center")

    plt.show()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from datetime import datetime

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 7)
###what is to_categorical doing here ? Question for you !

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape= (784,) ))
model.add(Dropout(0.2)) ###using dropout
model.add(Dense(256,activation='relu', kernel_regularizer = 'l2'))###using regularizer
model.add(Dropout(0.2))###using dropout
model.add(Dense(256,activation='relu', kernel_regularizer = 'l2')) ###using regularizer
model.add(Dropout(0.2))###using dropout
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
training_history = model.fit(
X_train, # input
y_train, # output
batch_size=32,
verbose=1, # Suppress chatty output; use Tensorboard instead
epochs=10,
validation_data=(X_test, y_test),
callbacks=[tensorboard_callback, es],
)

