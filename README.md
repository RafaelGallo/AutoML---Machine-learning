# AutoML Machine learning


Projeto voltado para modelos de automl com AutoKeras, H2O, Auto-Sklearn, AdaNet


[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/AutoKeras-red.svg)](https://autokeras.com/)
[![](https://img.shields.io/badge/H2O-yellow.svg)](https://www.h2o.ai/pt/)
[![](https://img.shields.io/badge/Auto_Sklearn-green.svg)](https://automl.github.io/auto-sklearn/master/)
[![](https://img.shields.io/badge/AdaNet-orange.svg)](https://github.com/tensorflow/adanet)


![Logo](https://github.com/RafaelGallo/AutoML---Machine-learning/blob/main/857.jpg)

# Versão das bibliotecas

| Nome             | Versão                                                                |
| ----------------- | ------------------------------------------------------------------ |
| AdaNet |  Versão: 0.9.0 |
| Auto-Keras | Versão: 1.0.16|
| H2O | Versão: 3.34.0.3|
| Auto-Sklearn | Versão: 0.14.2|


# Documentação 

Auto-Keras [Documentação](https://autokeras.com/)

Auto-Sklearn [Documentação](https://automl.github.io/auto-sklearn/master/api.html)

Auto-H2O [Documentação](https://docs.h2o.ai/)

Auto-AdaNet [Documentação](https://adanet.readthedocs.io/en/v0.9.0/)


# Instalação

Instalação das bibliotecas do AutoML.
Pode usar o comando pip para instalar as bibliotecas o pacote oficial do PyPi:

AutoKeras

```bash
  pip install autokeras
```

H20

```bash
  pip install h2o
```


Auto-Sklearn


```bash
  pip install auto-sklearn
```

AdaNet

```bash
  pip install adanet
```
# Demo
Exemplo simples como é AutoML para modelos de classificação e regressão linear.

```bash
  # Carregando as bibliotecas 
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Carregando o dataset
  data = pd.read_csv("data.csv")
  
  # Visualizando os 5 primeiros itens
  data.head()

  # visualizando linhas e colunas com shape
  data.shape

  # Informações das variaveis
  data.info()

  # Treino e teste da base de dados para x e y
  x = df_train.iloc[:, 0: 10]
  y = df_train.iloc[:, 10]

  # Visualizando o shape da variavel x
  x.shape

  # Visualizando o shape da variavel y
  y.shape

  # Treinando modelo de machine learning
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

  # Visualizando linhas e colunas do dado de treino x_train
  x_train.shape

  # Visualizando linhas e colunas do dado de treino y_train
  y_train.shape

  # Modelo AutoML - AutoKeras
  # Modelo classificação com autokeras

  # Importando a biblioteca 
  import autokeras as ak
  
  # Modelo de classificação - max_trials tempo que vai ser treinado
  model = ak.StructuredDataClassifier(max_trials = 10)
  
  # Modelo vai treino - epochs sera as epocas do modelo
  model_fit = model.fit(x = x_train, y = y_train, epochs = 100)
  
  # Valição do modelo foi preparado
  model_automl_eval = model.evaluate(x = x_test, y = y_test)

  # Previsão do modelo
  predict = model.predict(model_predict)
  predict
  
 
