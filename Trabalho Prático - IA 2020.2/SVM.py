import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import sklearn.svm
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.datasets import make_blobs

filename = input('> Digite o nome do arquivo: (lembre-se das extensões .csv ou .data)')

dataset = pd.read_csv(filename)
print(dataset.shape)
#dataset.describe().transpose()

if filename == 'zoo_classification.csv':
  dataset = dataset.drop('animal_name', axis = 1)
if filename == 'breast_cancer.csv':
    dataset = dataset.drop('id', axis=1)

median = dataset.median()
dataset.fillna(median, inplace=True)

X = pd.DataFrame(dataset.iloc[:,:-1])
target = dataset[dataset.columns[-1]]
y = target # Dividindo e armazenando os conjuntos de atributos dependentes e independentes


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=109) # 80% treino and 20% testes
classifier = SVC(kernel='linear')  # Criando um classificador SVM

classifier.fit(X_train, y_train)  # Treinando do modelo com os conjuntos de treinamento

y_pred = classifier.predict(X_test) # Predição para o dataset

print(f'Classification report: {classification_report(y_test, y_pred)}')
print(f'Precisão/Score: {classifier.score(X_test, y_pred)}')
print(f'Acurácia: {metrics.accuracy_score(y_test, y_pred)}')

scores = cross_val_score(classifier, X, y, cv=10)
print(f'Cross-validation score: {scores.mean()}')


cm = confusion_matrix(y_test, y_pred)
print(f'Confusion matrix: {cm}')


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
matrix_df = pd.DataFrame(confusion_matrix)
ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10,7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize =15)
plt.show()