import os
import numpy as np
import cv2
import tqdm
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,BatchNormalization,Flatten,AvgPool2D
import kerastuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
#from sklearn.model_selection import train_test_split
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


#from sklearn.model_selection import cross_val_score


# Setting the image size, epochs, classes and batch size
img=150
epochs=30
batch_size=25
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
                #r, c = image.shape
                #image = image.astype('uint8')
                
                image=cv2.resize(image, (img, img)) # Resizing to 150x150
            
                img_to_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # Histogram Equalization
                img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
                image = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
                
                #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscaling c/ problema
                
                images.append(image)
                labels.append(label)
                
        images=np.array(images,dtype=np.float32)
        labels=np.array(labels,dtype=np.int32)
        output.append((images,labels))
        
    return output

# Loading the training and testing data
print("Pre-processing...")
(train_images,train_labels),(test_images,test_labels)=load_data()

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

#--------------------------------------------------------------------------------------------------
# Making the model architecture for TensorFlow
print("Evaluating TF model...")

model=Sequential([
    Conv2D(filters=32,activation='relu',input_shape=(img,img,3),padding='same',kernel_size=(3,3)),
    Conv2D(filters=32,activation='relu',padding='same',kernel_size=(3,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64,activation='relu',padding='same',kernel_size=(3,3)),
    Conv2D(filters=64,activation='relu',padding='same',kernel_size=(3,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=128,activation='relu',padding='same',kernel_size=(3,3)),
    Conv2D(filters=128,activation='relu',padding='same',kernel_size=(3,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=256,activation='relu',padding='same',kernel_size=(3,3)),
    Conv2D(filters=256,activation='relu',padding='same',kernel_size=(3,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=256,activation='relu',padding='same',kernel_size=(3,3)),
    Conv2D(filters=256,activation='relu',padding='same',kernel_size=(3,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=512,activation='relu',padding='same',kernel_size=(3,3)),
    Conv2D(filters=512,activation='relu',padding='same',kernel_size=(3,3)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=512,activation='relu',padding='same',kernel_size=(3,3)),
    Conv2D(filters=512,activation='relu',padding='same',kernel_size=(3,3)),
    Flatten(),
    Dense(units=4096,activation='relu'),
    Dense(units=4096,activation='relu'),
    Dense(units=1,activation='sigmoid'),
])


# Printing the model summary
model.summary()

# Saving the weights of the model
model.save('model.h5')

# Compiling the model. Specifying the optimizer to act on the data. The loss function(which could also have been sparse_categorical_crossentropy),since there are only two classes, I have used binary_crossentropy
model.compile(optimizer=Adam(lr=0.0001),loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('model.h5')

#-----------------------------------------------------------------
# TensorFlow

# Training the model on the training data
#print("Fitting...")
#model.fit(x=train_images, y=train_labels)
#, validation_split=0.3,epochs=epochs,batch_size=batch_size,steps_per_epoch=100,verbose=2
#model.fit(x=train_images,y=train_labels,validation_split=0.3,epochs=epochs,batch_size=batch_size,steps_per_epoch=100,verbose=2)
# Using the model to make predictions on the testing data
print("Predicting...")
y_pred = model.predict(test_images)

print('Done.')

# Using the confusion matrix to print the accuracy of those predictions
cm = confusion_matrix(y_true=test_labels,y_pred=np.round(y_pred))
print(f'\nAccuracy Score based on Confusion Matrix (TF): {(cm.trace()/cm.sum())*100}%')

scores = cross_val_score(model, train_images, train_labels, cv=10)
print(f'Cross-validation score: {scores.mean()}')
#print('\nClassification report: ')
#print(classification_report(test_labels, y_pred))