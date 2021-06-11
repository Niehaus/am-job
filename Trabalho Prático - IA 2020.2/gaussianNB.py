import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion matrix: {cm}')
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
print(f'Score: {classifier.score(X_test, predictions)}')
print(f'Accuracy: {metrics.accuracy_score(y_test, predictions)}')

scores = cross_val_score(classifier, X, y, cv=10)
print(f'Cross-validation score: {scores.mean()}')

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
matrix_df = pd.DataFrame(confusion_matrix)
ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10,7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize =15)
plt.show()