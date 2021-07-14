import os
import numpy as np
import cv2
import tqdm
import random
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Setting the image size, epochs, classes and batch size.
img = 150
epochs = 30
batch_size = 25
names=['O','R']
encode_name={name:i for i,name in enumerate(names)}

# Making a function to load the images and labels. OpenCV was used for this purpose
# Image resizing and data augmentation is happening in this module.
def load_data():
    datasets=['DATASET/TRAIN','DATASET/TEST']
    output=[]
    for dataset in tqdm.tqdm(datasets):
        images=[]
        labels=[]
        for folder in os.listdir(dataset):
            label=encode_name[folder]
            if dataset=='DATASET/TRAIN':
                img_set=random.sample(os.listdir(os.path.join(dataset,folder)), 7000)
            else:
                img_set=random.sample(os.listdir(os.path.join(dataset,folder)), 1000)
            for file in img_set:
                img_path=os.path.join(os.path.join(dataset, folder), file)
                image=cv2.imread(img_path)
                
                image=cv2.resize(image, (img, img)) # Resizing to 150x150
            
                img_to_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # Histogram Equalization
                img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
                image = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscaling
                
                images.append(image.flatten()) # Flattening
                labels.append(label)
                
        images=np.array(images,dtype=np.float32)
        labels=np.array(labels,dtype=np.int32)
        output.append((images,labels))
        
    return output


# Loading the training and testing data
print("Pre-processing...")
(train_images,train_labels),(test_images,test_labels) = load_data()

# Converting the training and testing images and labels to numpy arrays
train_images=np.array(train_images)
train_labels=np.array(train_labels)

test_images=np.array(test_images)
test_labels=np.array(test_labels)

#Printing the size of the images and labels
#print(f'Tamanho das magens de treino:{train_images.shape}, Tamanho dos rótulos de teste:{train_labels.shape}')
#print(f'Tamanho das magens de teste:{test_images.shape}, Tamanho dos rótulos de teste:{test_labels.shape}')

#Scaling the values of the images pixels to 0-1 to make the computation easier for our model
train_images,test_images=train_images/255,test_images/255


#Randomizing the training data
train_images,train_labels=shuffle(train_images,train_labels)
#le = LabelEncoder()

#-------------------------------------------------------------------------
# "Training" and evaluating the KNN Classifier on the raw pixel intensities
print("Evaluating KNN...")
model = KNeighborsClassifier(n_neighbors = 10, n_jobs=-1)

print("Fitting...")
model.fit(train_images, train_labels)

print("Predicting...")
y_pred = model.predict(test_images)

print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels,y_pred=np.round(y_pred))

print(f'\nAccuracy Score based on Confusion Matrix (KNN): {(cm.trace()/cm.sum())*100}%')
cmKNN=confusion_matrix(y_true=test_labels,y_pred=np.round(y_pred))

print('\nClassification Report:')
print(classification_report(test_labels, y_pred))

scores = cross_val_score(model, train_images, train_labels, cv=10)
print(f'Cross-validation score: {scores.mean()}')