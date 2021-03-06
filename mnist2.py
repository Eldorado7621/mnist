# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 00:38:26 2021

@author: akino

 we will classify handwritten digits using a simple neural network which has only input and
 output layers. We will then add a hidden layer and see how the performance of the model
 improves
 
 MNIST (Modified National Institute of Standards and Technology database) is a large database
 of 70,000 handwritten digits.
"""

import tensorflow as tf # deep learning library
import numpy as np # for matrix operations
import matplotlib.pyplot as plt # for visualization
from tensorflow.keras.datasets.mnist import load_data # To load the MNIST digit dataset

(X_train, y_train) , (X_test, y_test) = load_data() # Loading data
#plt.matshow(X_train[0])

# code to view the images
num_rows, num_cols = 2, 5
f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),
gridspec_kw={'wspace':0.03, 'hspace':0.01},
squeeze=True)

for r in range(num_rows):
    for c in range(num_cols):

            image_index = r * 5 + c
            ax[r,c].axis("off")
            ax[r,c].imshow( X_train[image_index], cmap='gray')
            ax[r,c].set_title('No. %d' % y_train[image_index])
plt.show()
plt.close()

#normalize the data
X_train = X_train / 255
X_test = X_test / 255
#flatten the data
X_train_flattened = X_train.reshape(len(X_train), 28*28) 
# converting our 2D array representin an image to one dimensional
X_test_flattened = X_test.reshape(len(X_test), 28*28)

#building the model
#first we build a simple neural model with  hidden layers

# Defining the model
model = tf.keras.Sequential([
tf.keras.layers.Dense(100, input_shape=(784,), activation='relu'),
tf.keras.layers.Dense(100, input_shape=(100,),activation='relu'),
tf.keras.layers.Dense(10, activation='sigmoid')
])
model.summary()
# Compiling the model
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# Fit the model
model.fit(X_train_flattened, y_train, batch_size= 128,epochs=5)

# Evaluate the model
model.evaluate(X_test_flattened,y_test)
# saving the model
save_dir = "/results/"
model_name = 'keras_mnist.h5'
model.save(model_name)
model_path = save_dir + model_name
print('Saved trained model at %s ' % model_path)
