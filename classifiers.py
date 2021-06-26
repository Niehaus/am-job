from sklearn.svm import SVC
from abc import ABC
from utils.utils import *
from sklearn.metrics import confusion_matrix


class Classifiers(ABC):
    def __init__(self, classifier, name):
        self.name = name
        self.classifier = classifier

    @property
    def classifier_name(self):
        return self.name


class SVM(Classifiers):
    def fit(self, images, labels):
        self.classifier.fit(images, labels)

    def predict(self, images):
        return self.classifier.predict(images)

    def score(self, test_images, y_pred):
        return self.classifier.score(test_images, y_pred)


class TensorFlow(Classifiers):
    def fit(self, images, labels,):
        self.classifier.fit(
            x=images, y=labels, validation_split=0.3,
            epochs=epochs, batch_size=batch_size,
            steps_per_epoch=100, verbose=2
        )

    def predict(self, images):
        return self.classifier(images)

    @staticmethod
    def score(labels, pred):
        cm = confusion_matrix(y_true=labels, y_pred=np.round(pred))
        return (cm.trace() / cm.sum()) * 100

class NaiveBayes(Classifiers):
    ...
