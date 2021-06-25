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

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

img=150
epochs=30
batch_size=25
names=['O','R']
encode_name={name:i for i,name in enumerate(names)}

def load_data():
    datasets=['DATASET/TRAIN','DATASET/TEST']
    output=[]
    for dataset in tqdm.tqdm(datasets):
        images=[]
        labels=[]
        for folder in os.listdir(dataset):
            label=encode_name[folder]
            if dataset=='DATASET/TRAIN':
                img_set=random.sample(os.listdir(os.path.join(dataset,folder)),7000)
            else:
                img_set=random.sample(os.listdir(os.path.join(dataset,folder)),1000)
            for file in img_set:
                img_path=os.path.join(os.path.join(dataset,folder),file)
                image=cv2.imread(img_path)
                image=cv2.resize(image,(img,img))
                images.append(image)
                labels.append(label)
        images=np.array(images,dtype=np.float32)
        labels=np.array(labels,dtype=np.int32)
        output.append((images,labels))
    return output

(train_images,train_labels),(test_images,test_labels)=load_data()

train_images=np.array(train_images)
train_labels=np.array(train_labels)

test_images=np.array(test_images)
test_labels=np.array(test_labels)

print(f'Tamanho das magens de treino:{train_images.shape}, Tamanho dos rótulos de teste:{train_labels.shape}')
print(f'Tamanho das magens de teste:{test_images.shape}, Tamanho dos rótulos de teste:{test_labels.shape}')

train_images,test_images=train_images/255,test_images/255

train_images,train_labels=shuffle(train_images,train_labels)

#-----------------------------
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

model.summary()

model.save('model.h5')

model.compile(optimizer=Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

model.load_weights('model.h5')
#-----------------------
# SVM Classifier
classifier = SVC(kernel='rbf')  # Criando um classificador SVM

classifier.fit(train_images, train_labels)  # Treinando do modelo com os conjuntos de treinamento

y_pred = classifier.predict(test_images) # Predição  para o dataset

print(f'Classification report (SVM): {classification_report(test_labels, y_pred)}')
print(f'Precisão/Score (SVM): {classifier.score(test_images, y_pred)}')
print(f'Acurácia(SVM): {metrics.accuracy_score(test_labels, y_pred)}')

#scores = cross_val_score(classifier, X, y, cv=10)
#print(f'Cross-validation score: {scores.mean()}')

#------------------
# TensorFlow

model.fit(x=train_images,y=train_labels,validation_split=0.3,epochs=epochs,batch_size=batch_size,steps_per_epoch=100,verbose=2)

p=model.predict(test_images)

cm=confusion_matrix(y_true=test_labels,y_pred=np.round(p))
print(f'Accuracy (TENSORFLOW):{(cm.trace()/cm.sum())*100}')
