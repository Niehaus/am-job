#!/usr/bin/env python
# coding: utf-8

# # Trabalho Prático 1

# In[2]:


import os
import numpy as np
import cv2
import tqdm
import random
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import tree
from utils.utils import load_data


# ### Pré-processamento das Imagens

# In[3]:


# Loading the training and testing data
# print("Pre-processing...")
(train_images, train_labels), (test_images, test_labels) = load_data()

# Converting the training and testing images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Scaling the values of the images pixels to 0-1 to make the computation easier for our model
train_images, test_images = train_images / 255, test_images / 255

# Randomizing the training data
train_images, train_labels = shuffle(train_images, train_labels)


# ### Decision Tree Classifier

# In[ ]:


# Training and evaluating the Decision Tree Classifier on the raw pixel intensities
#print("Evaluating DT...")
classifier = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)

# print("Fitting...")
classifier.fit(train_images, train_labels)

# print("Predicting...")
y_pred = classifier.predict(test_images)

# print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels, y_pred=np.round(y_pred))
# print(f'\nAccuracy Score Confusion Matrix (DT): {(cm.trace() / cm.sum()) * 100}%')

# print('\nClassification Report:')
# print(classification_report(test_labels, y_pred))
# print(classifier.score(test_images, y_pred))

scores = cross_val_score(classifier, train_images, train_labels, cv=10)
print(f'Decisione Tree Cross-validation score: {scores.mean()}')


# ### KNN

# In[4]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# "Training" and evaluating the KNN Classifier on the raw pixel intensities
#print("Evaluating KNN...")
model = KNeighborsClassifier(n_neighbors = 10, n_jobs=-1)

#print("Fitting...")
model.fit(train_images, train_labels)

#print("Predicting...")
y_pred = model.predict(test_images)

#print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels,y_pred=np.round(y_pred))

#print(f'\nAccuracy Score based on Confusion Matrix (KNN): {(cm.trace()/cm.sum())*100}%')
cmKNN=confusion_matrix(y_true=test_labels,y_pred=np.round(y_pred))

#print('\nClassification Report:')
#print(classification_report(test_labels, y_pred))

scores = cross_val_score(model, train_images, train_labels, cv=10)
print(f'KNN Cross-validation score: {scores.mean()}')


# ### GAUSSIAN-NB Classifier

# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Training and evaluating classifier on the raw pixel intensities
# print("Evaluating GaussianNB...")

sc = StandardScaler()
train_images = sc.fit_transform(train_images)
test_images = sc.transform(test_images)

classifier = GaussianNB()

#print("Fitting...")
classifier.fit(train_images, train_labels)

#print("Predicting...")
y_pred = classifier.predict(test_images)

#print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels,y_pred=np.round(y_pred))
#print(f'\nAccuracy Score Confusion Matrix (N-Bayes): {(cm.trace()/cm.sum())*100}%')

#print('\nClassification Report:')
#print(classification_report(test_labels, y_pred))
#print(f'Score: {classifier.score(test_images, y_pred)}')

scores = cross_val_score(classifier, train_images, train_labels, cv=10)
print(f'GaussianN Cross-validation score: {scores.mean()}')


# ### SVM Classifier

# In[6]:


from sklearn.svm import SVC


# In[ ]:


# Training and evaluating the SVM Classifier on the raw pixel intensities
# print("Evaluating SVM...")
classifier = SVC(kernel='rbf')  # Creating a SVM Classifier

# print("Fitting...")
classifier.fit(train_images, train_labels)  # Model training with training set

# print("Predicting...")
y_pred = classifier.predict(test_images) # Model predicting

# print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels,y_pred=np.round(y_pred))
# print(f'\nAccuracy Score Confusion Matrix (SVM): {(cm.trace()/cm.sum())*100}%')

# print('\nClassification report: ')
# print(classification_report(test_labels, y_pred))

scores = cross_val_score(classifier, train_images, train_labels, cv=10)
print(f'SVM Cross-validation score: {scores.mean()}')


# ### TensorFlow 

# In[ ]:


from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from utils.utils import load_data_tensorflow


# In[ ]:


# Setting the image size, epochs, classes and batch size
img = 150
names = ['O', 'R']
encode_name = {name: i for i, name in enumerate(names)}

# Loading the training and testing data
# print("Pre-processing...")
(train_images, train_labels), (test_images, test_labels) = load_data_tensorflow()

# Converting the training and testing images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Scaling the values of the images pixels to 0-1 to make the computation easier for our model
train_images, test_images = train_images / 255, test_images / 255

# Randomizing the training data
train_images, train_labels = shuffle(train_images, train_labels)


# In[ ]:


# print("Evaluating TF model...")

model = Sequential([
    Conv2D(filters=32, activation='relu', input_shape=(img, img, 3), padding='same', kernel_size=(3, 3)),
    Conv2D(filters=32, activation='relu', padding='same', kernel_size=(3, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=64, activation='relu', padding='same', kernel_size=(3, 3)),
    Conv2D(filters=64, activation='relu', padding='same', kernel_size=(3, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=128, activation='relu', padding='same', kernel_size=(3, 3)),
    Conv2D(filters=128, activation='relu', padding='same', kernel_size=(3, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
    Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
    Conv2D(filters=256, activation='relu', padding='same', kernel_size=(3, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
    Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
    Conv2D(filters=512, activation='relu', padding='same', kernel_size=(3, 3)),
    Flatten(),
    Dense(units=4096, activation='relu'),
    Dense(units=4096, activation='relu'),
    Dense(units=1, activation='sigmoid'),
])

# Printing the model summary
model.summary()

# Saving the weights of the model
model.save('model.h5')

# Compiling the model. Specifying the optimizer to act on the data. 
# The loss function (which could also have been sparse_categorical_crossentropy),
# since there are only two classes, I have used binary_crossentropy
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('model.h5')


# In[ ]:


# print("Predicting...")
y_pred = model.predict(test_images)

# print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels, y_pred=np.round(y_pred))
# print(f'\nAccuracy Score based on Confusion Matrix (TF): {(cm.trace() / cm.sum()) * 100}%')

scores = cross_val_score(model, train_images, train_labels, cv=10)
print(f'TensorFlow Cross-validation score: {scores.mean()}')

