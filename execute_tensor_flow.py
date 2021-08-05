#!/usr/bin/env python
# coding: utf-8

# # Trabalho Prático 1

import os
import numpy as np
import cv2
import tqdm
import random
from sklearn.utils import shuffle
import time


# ### Pré-processamento das Imagens

from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from utils.utils import load_data_tensorflow
from tensorflow.keras import datasets, layers, models

# Alterar dados do modelo.
# Alterar algoritmo da predição.

img = 150
names = ['O', 'R']
encode_name = {name: i for i, name in enumerate(names)}
epochs = 30
batch_size = 25

(train_images, train_labels), (test_images, test_labels) = load_data_tensorflow()

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images, test_images = train_images / 255, test_images / 255
train_images, train_labels = shuffle(train_images, train_labels)

model = models.Sequential(name='CNN-CIFAR10')
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.save('model.h5')
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('model.h5')

model.fit(x=train_images,y=train_labels,validation_split=0.3,epochs=epochs,batch_size=batch_size,steps_per_epoch=100,verbose=2)
y_pred = model.predict(test_images)

cm = confusion_matrix(y_true=test_labels, y_pred=np.round(y_pred))
print(f'\nAccuracy Score based on Confusion Matrix (TF): {(cm.trace() / cm.sum()) * 100}%')