from random import shuffle
from csv import reader
import numpy as np
from numpy import log2
import math
import csv
import matplotlib as plt
import random
from random import randrange
import pandas as pd
import operator
from mlxtend.plotting import plot_decision_regions
from sklearn import linear_model

'''
Descrição:
	. split_Dataset(): Tratamento dos dados. Lê o arquivo e divide o dataset em conjunto de treinamento e conjunto de testes.
Entradas: 
	. filename: nome do arquivo (.csv) passado como parâmetro para o programa ao iniciar.
Resultado: 
	. train_X: Conjunto treino, test_X: Conjunto teste. 
'''
def split_Dataset(filename):
  df = pd.read_csv(filename)

  dataset = pd.DataFrame(df)

  median = dataset.median()
  dataset.fillna(median, inplace=True)

  if filename == 'breast_cancer.csv':
    dataset = dataset.drop('id', axis=1)
  if filename == 'zoo_classification.csv':
    dataset = dataset.drop('animal_name', axis = 1)

  dataset = dataset.sample(frac=1).reset_index(drop=True)

  features = dataset.iloc[:,:1]  # Armazenando os n-1 atributos
  labels = dataset.iloc[:,-1] # Armazenando a coluna classe
  
 # dataset.fillna(features.median(), inplace = True) # Preenchendo os espaços vazios das colunas (nesse caso foi necessário para o dataset breastCancer.csv, pois não sabia que possuia valores faltantes)

  split_porcentage = int(0.8 * len(dataset))  # 80% do conjunto de dados será utilizado para treinamento. 20% para testes.
  #print(f'\nSplit Porcentage: {split_porcentage}')

  train_X = dataset[:split_porcentage]
  test_X = dataset[split_porcentage:]

  train_X = list(train_X.values)
  test_X = list(test_X.values)

  return train_X, test_X
  
'''
Descrição:
	. def distancia_euclidiana(): Função para cálculo da distância euclidiana.
Entradas: 
	. instância x e instância y: Vetores de entrada para os quais a distância será calculada.
Resultado: 
	. Distância euclidiana calculada a partir dos dois vetores.
'''
def distancia_euclidiana(x, y):

  distancia = 0.0
  for i in range(len(x)-1): # A distância é calculada entre todos os vetores do conjunto, exceto a a classe.
    distancia += math.pow(x[i] - y[i], 2)
    #print(f'Distanciaeuclidy: {distancia}')  
  raiz = math.sqrt(distancia)
  
  return raiz

'''
Descrição: 
	. def vizinhos_proximos(): Definindo os k-vizinhos mais próximos de uma instância com base nas distâncias.
Entradas:
	. conj_treino: Conjunto de dados de treino;
	. instância: Instância do conjunto sendo analisada;
	. k-vizinhos: Número de vizinhos próximos a serem retornados;
Resultado:
	. Os k-vizinhos mais próximos à instância analisada.
'''
def vizinhos_proximos(conj_treino, instancia, k_vizinhos):
  distancias = list()
  #for i in range(len(conj_treino)):
  for instancia_treino in conj_treino:
    distancia = distancia_euclidiana(instancia, instancia_treino)
    distancias.append((instancia_treino, distancia))

  distancias.sort(key=operator.itemgetter(1)) # Ordenando por proximidade

  vizinhos_proximos = []

  for i in range(k_vizinhos):  # Obtendo apenas o k-vizinhos mais próximos
    vizinhos_proximos.append(distancias[i][0])
	
  return vizinhos_proximos

'''
Descrição:
	. def classificacao(): Aplicar o KNN ao dado/vetor analisado
Entradas:
	. conjunto_treino: Conjunto de dados de treino;
	. instancia: instância do conjunto sendo analisada;
	. k-vizinhos: k-vizinhos próximos de interesse;
Resultado
	. Predição sobre qual classe o vetor analisado pertence.
'''
def classificacao(conjunto_treino, instancia, k_neighbors):
	vizinhos_prox = vizinhos_proximos(conjunto_treino, instancia, k_neighbors)
	resultado = [row[-1] for row in vizinhos_prox]
	predicao = max(set(resultado), key=resultado.count)

	#predictions = []
	#predictions.append(prediction)
  
	return predicao

def k_vizinhos_proximos(conjunto_treino, k_vizinhos):
	predicoes = []
	for instancia in conjunto_treino:
		saida = classificacao(conjunto_treino, instancia, k_vizinhos)
		predicoes.append(saida)
	return (predicoes)
#########################################################################

'''
Descrição: acuracia():  % de acertos do algoritmo sobre o conjunto de dados.

Entradas: conjunto_teste, predicoes

Resultado: Taxa de acuracidade.
'''
def acuracia(conjunto_teste, predicoes):
    acertos = 0
    for x in range(len(conjunto_teste)):
        if conjunto_teste[x][-1] == predicoes[x]:
            acertos += 1

    acuracia = acertos/float(len(conjunto_teste))
    return acuracia

filename = input('Digite o nome do arquivo (.csv):')

train_X, test_X = split_Dataset(filename)

#k_folds = 10
k_neighbors = 10

predicao = k_vizinhos_proximos(train_X, k_neighbors)

acuracia = acuracia(test_X, predicao)
print(f'Acurácia: {acuracia}%')