import os
import numpy as np
import cv2
import tqdm
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

from utils.utils import *
from classifiers import SVM, TensorFlow


(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, train_labels = train(train_images, train_labels)
test_images, test_labels = train(test_images, test_labels)

print(f'Tamanho das magens de treino:{train_images.shape}, '
      f'Tamanho dos rótulos de teste:{train_labels.shape}')

print(f'Tamanho das magens de teste:{test_images.shape}, '
      f'Tamanho dos rótulos de teste:{test_labels.shape}')

train_images, test_images = train_images / 255, test_images / 255
train_images, train_labels = shuffle(train_images, train_labels)

# -----------------------------
'''
model = Sequential()
model.add(Conv2D(16,(5,5),padding='valid',input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.4))
model.add(Conv2D(32,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.6))
model.add(Conv2D(64,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)
feat_train = model_feat.predict(X_train)
'''
#
# model = Sequential([
#     Conv2D(filters=32, activation='relu', input_shape=(img, img, 3), padding='same', kernel_size=(3, 3)),
#     Conv2D(filters=32, activation='relu', padding='same', kernel_size=(3, 3)),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(filters=64, activation='relu', padding='same', kernel_size=(3, 3)),
#     Conv2D(filters=64, activation='relu', padding='same', kernel_size=(3, 3)),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(filters=128, activation='relu', padding='same', kernel_size=(3, 3)),
#     Conv2D(filters=128, activation='relu', padding='same', kernel_size=(3, 3)),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
#     Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
#     Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
#     Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
#     Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
#     Flatten(),
#     Dense(units=4096, activation='relu'),
#     Dense(units=4096, activation='relu'),
#     Dense(units=1, activation='sigmoid'),
# ])

# model.summary()
# model.save('model.h5')
# model.compile(
#     optimizer=Adam(lr=0.0001),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# model.load_weights('model.h5')

# -----------------------
# SVM Classifier
svm = SVM(SVC(kernel='rbf'), "SVM")

svm.fit(train_images, train_labels)
y_pred = svm.predict(test_images)

print(f'Classification report (SVM): {classification_report(test_labels, y_pred)}')
print(f'Precisão/Score (SVM): {svm.score(test_images, y_pred)}')
print(f'Acurácia(SVM): {metrics.accuracy_score(test_labels, y_pred)}')

# scores = cross_val_score(classifier, X, y, cv=10)
# print(f'Cross-validation score: {scores.mean()}')

# ------------------
# TensorFlow
# tensor_flow = TensorFlow(model, "Tensor Flow")
# tensor_flow.fit(train_images, train_labels)
# tf_predict_score = tensor_flow.predict(test_images)
#
# accuracy = tensor_flow.score(test_labels, np.round(tf_predict_score))
# print(f'Accuracy {tensor_flow.classifier_name}:{accuracy}')
