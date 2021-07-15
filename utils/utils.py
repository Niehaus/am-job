import numpy as np
import cv2
import tqdm
import os
import random

img = 150
epochs = 30
batch_size = 25
names = ['O', 'R']
encode_name = {name: i for i, name in enumerate(names)}


def load_data():
    datasets = ['DATASET/TRAIN', 'DATASET/TEST']
    output = []
    for dataset in tqdm.tqdm(datasets):
        images = []
        labels = []
        for folder in os.listdir(dataset):
            label = encode_name[folder]
            if dataset == 'DATASET/TRAIN':
                img_set = random.sample(os.listdir(os.path.join(dataset, folder)), 7000)
            else:
                img_set = random.sample(os.listdir(os.path.join(dataset, folder)), 1000)
            for file in img_set:
                img_path = os.path.join(os.path.join(dataset, folder), file)
                image = cv2.imread(img_path)

                image = cv2.resize(image, (img, img))  # Resizing to 150x150

                img_to_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Histogram Equalization
                img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
                image = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscaling

                images.append(image.flatten())  # Flattening
                labels.append(label)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        output.append((images, labels))

    return output


def load_data_tensorflow():
    datasets = ['DATASET/TRAIN', 'DATASET/TEST']
    output = []
    for dataset in tqdm.tqdm(datasets):
        images = []
        labels = []
        for folder in os.listdir(dataset):
            label = encode_name[folder]
            if dataset == 'DATASET/TRAIN':
                img_set = random.sample(os.listdir(os.path.join(dataset, folder)), 7000)
            else:
                img_set = random.sample(os.listdir(os.path.join(dataset, folder)), 1000)
            for file in img_set:
                img_path = os.path.join(os.path.join(dataset, folder), file)
                image = cv2.imread(img_path)
                # r, c = image.shape
                # image = image.astype('uint8')

                image = cv2.resize(image, (img, img))  # Resizing to 150x150

                img_to_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Histogram Equalization
                img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
                image = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

                # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscaling c/ problema

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        output.append((images, labels))

    return output


def train(images, labels):
    train_images = np.array(images)
    train_labels = np.array(labels)

    return train_images, train_labels
