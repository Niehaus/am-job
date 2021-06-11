from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

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
#target = lympho_dataset.iloc[:,-1:]
target = dataset[dataset.columns[-1]]
Y = target # Dividindo e armazenando os conjuntos de atributos dependentes e independentes

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.2)
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),activation="relu",random_state=1).fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')

fig=plot_confusion_matrix(clf, X_testscaled, y_test)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()