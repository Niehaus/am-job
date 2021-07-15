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

# from sklearn import svm


# Setting the image size, epochs, classes and batch size
img = 150
epochs = 30
batch_size = 25
names = ['O', 'R']
encode_name = {name: i for i, name in enumerate(names)}

# Loading the training and testing data
print("Pre-processing...")
(train_images, train_labels), (test_images, test_labels) = load_data()

# Converting the training and testing images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Printing the size of the images and labels
# print(f'Tamanho das magens de treino:{train_images.shape}, Tamanho dos rótulos de teste:{train_labels.shape}')
# print(f'Tamanho das magens de teste:{test_images.shape}, Tamanho dos rótulos de teste:{test_labels.shape}')

# Scaling the values of the images pixels to 0-1 to make the computation easier for our model
train_images, test_images = train_images / 255, test_images / 255

# Randomizing the training data
train_images, train_labels = shuffle(train_images, train_labels)

# ---------------------------------------------------------------------------------------------
# DECISION TREE CLASSIFIER

# Training and evaluating the Decision Tree Classifier on the raw pixel intensities
print("Evaluating DT...")
classifier = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)

print("Fitting...")
classifier.fit(train_images, train_labels)

print("Predicting...")
y_pred = classifier.predict(test_images)

print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels, y_pred=np.round(y_pred))
print(f'\nAccuracy Score Confusion Matrix (DT): {(cm.trace() / cm.sum()) * 100}%')

print('\nClassification Report:')
print(classification_report(test_labels, y_pred))
print(classifier.score(test_images, y_pred))

scores = cross_val_score(classifier, train_images, train_labels, cv=10)
print(f'Cross-validation score: {scores.mean()}')
