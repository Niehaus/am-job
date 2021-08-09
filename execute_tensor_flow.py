#!/usr/bin/env python
# coding: utf-8

# # Trabalho Prático 1

import os
import numpy as np
import cv2
import tqdm
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from utils.utils import load_data_tensorflow
from tensorflow.keras import datasets, layers, models
from keras.losses import SparseCategoricalCrossentropy

# Alterar dados do modelo.
# Alterar algoritmo da predição.

def generate_model():
    model = models.Sequential(name='CNN-WASTE-CLASSIFICATION')
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img, img, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model

img = 150
names = ['O', 'R']
encode_name = {name: i for i, name in enumerate(names)}
epochs = 2
batch_size = 25

(train_images, train_labels), (test_images, test_labels) = load_data_tensorflow()

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images, test_images = train_images / 255, test_images / 255
train_images, train_labels = shuffle(train_images, train_labels)

model = generate_model()
model.summary()
model.save('model.h5')

# model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# model.load_weights('model.h5')

# Incluir variações nesse ponto
# Variar algoritmos e perda. A métrica sempre vai ser a acuracia.
"""
    compile(
        optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
        weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs
    )

"""
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1, 
                    validation_data=(test_images, test_labels))

y_pred = model.evaluate(test_images, test_labels, batch_size=128)
print("test loss, test acc:", y_pred)

# print(history.history['accuracy'] + 'Acurácia de Treino')
# print(history.history['val_accuracy'] + 'Acurácia de Validação')
# print(history.history['loss'] + 'Perda de Treino')
# print(history.history['val_loss'] + 'Perda de Validação')