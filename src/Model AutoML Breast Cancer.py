#!/usr/bin/env python
# coding: utf-8

# # Model Automl - Breast Cancer

# In[1]:


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Biblioteca AutoML Pycaret
from pycaret.classification import *

# Parâmetros de configuração dos gráficos
from matplotlib import rcParams

rcParams['figure.figsize'] = 12, 4
rcParams['lines.linewidth'] = 3
rcParams['xtick.labelsize'] = 'x-large'
rcParams['ytick.labelsize'] = 'x-large'

# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())

# Versões dos pacotes usados neste jupyter notebook
#%reload_ext watermark
#%watermark -a "Verções bibliotecas" --iversions

import warnings
warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_csv("data.csv")
data


# In[5]:


# Visualizando os 5 primeiros dados
data.head(5)


# In[6]:


# Visualizando os 5 últimos dados
data.tail(5)


# In[7]:


# Visualizando linhas e colunas
data.shape


# In[8]:


# Informações dos dados
data.info()


# In[9]:


# Tipo dados
data.dtypes


# In[10]:


# Amostra simples 5 
data.sample(5)


# In[11]:


# Colunas númericas
nub = ["perimeter_mean", 
       "area_mean", 
       "smoothness_mean", 
       "compactness_mean", 
       "concavity_mean",
       "concave points_mean", 
       "symmetry_mean", 
       "fractal_dimension_mean", 
       "radius_se", 
       "perimeter_se",
       "area_se",
       "smoothness_se"]

# Coluna target
target = ["diagnosis"]


# In[12]:


# Total
data["diagnosis"].value_counts()


# In[13]:


# Variavel target
target = ["diagnosis"]


# # Explorando as variáveis númericas

# In[14]:


# Resumo variáveis numéricas
data[nub].describe()


# In[15]:


fig , axes = plt.subplots(nrows=3,ncols=3 , figsize = (20, 22))

ax = sns.distplot(data["perimeter_mean"] , ax=axes[0, 0])
ax = sns.distplot(data["area_mean"] ,  ax=axes[0, 1])
ax = sns.distplot(data["smoothness_mean"] , ax=axes[0, 2])
ax = sns.distplot(data["compactness_mean"], ax=axes[1, 0] )
ax = sns.distplot(data["concavity_mean"] , ax=axes[1, 1] )
ax = sns.distplot(data["concave points_mean"] , ax=axes[1, 2] )
ax = sns.distplot(data["symmetry_mean"] , ax=axes[2, 0])
ax = sns.distplot(data["fractal_dimension_mean"], ax=axes[2, 1])
ax = sns.distplot(data["radius_se"], ax=axes[2, 2])


plt.show()


# In[16]:


fig,ax = plt.subplots(1,2,figsize=(13,5))
sns.boxplot(y=data['fractal_dimension_mean'],x=data['diagnosis'],ax=ax[0])
sns.boxplot(y=data['fractal_dimension_mean'],x=data['diagnosis'],ax=ax[1])
plt.tight_layout()


# In[17]:


# Correlação das colunas númericas
data[nub].corr()


# In[18]:


# Correlação
data_corr = data[nub].corr()
data_corr


# In[19]:


# Gráfico
plt.figure(figsize = (10.8, 8))
sns.heatmap(data_corr, cmap = 'Blues', annot = True, fmt = '.2f')


# In[20]:


plt.figure(figsize = (15, 15))
sns.pairplot(data[nub], diag_kind = 'kde')


# # Modelo AutoML

# In[32]:


# Nessa função ele cria o pipeline transformação modelo 
# Segunda parte setup deve ser chamada antes de executar para função.

# Model
# Raiz da base dados
model = setup(data = data,
      # Features target
      target = "diagnosis",

      # Os valores ausentes em recursos numéricos são imputados com o valor 'médio' do recurso no conjunto de dados de treinamento. 
      # A outra opção disponível é 'mediana' ou 'zero'.
      numeric_imputation = 'mean',

      # Controla a entrada de confirmação de tipos de dados quando setupé executado. 
      # Ao executar em modo totalmente automatizado ou em um kernel remoto, deve ser True.
      silent = True)


# In[33]:


# Essa função treina e avalia o desempenho de todos os estimadores disponíveis na biblioteca de modelos usando validação cruzada. 
# A saída dessa função é uma grade de pontuação com pontuações médias de validação cruzada. 
# As métricas avaliadas durante o CV podem ser acessadas usando a função get_metrics. 
# As métricas personalizadas podem ser adicionadas ou removidas usando as funções add_metric e remove_metric.
compare_models()


# In[34]:


gbc = create_model('gbc')


# In[35]:


Random_Forest = create_model('rf')


# In[36]:


Tuned_Random_Forest = tune_model(Random_Forest)


# In[37]:


plot_model(estimator = Tuned_Random_Forest, plot = 'learning')


# In[38]:


plot_model(estimator = Tuned_Random_Forest, plot = 'auc')


# In[39]:


plot_model(estimator = Tuned_Random_Forest, plot = 'confusion_matrix')


# In[40]:


plot_model( estimator = Tuned_Random_Forest, plot = 'feature')


# In[41]:


evaluate_model(Tuned_Random_Forest)


# In[42]:


interpret_model( Tuned_Random_Forest )


# # Modelo 02 - AutoML

# In[21]:


# Modelo automl 2
model = setup(data, 
             target = "diagnosis",
             session_id = 123, 
             log_experiment = True, 
             numeric_imputation = 'mean',
             silent = True)


# In[22]:


# Modelos melhores
model = compare_models()


# In[47]:


# Modelo Light Gradient Boosting Machine
lightgbm = create_model('lightgbm')


# In[48]:


# Model Gradient Boosting Classifier
gbc = create_model('gbc')


# In[49]:


# Model Ada Boost Classifier
ada = create_model('ada', fold = 10)


# In[50]:


# Tuned dos modelos

model_tuned_lr = tune_model(lightgbm)
model_tuned_rf = tune_model(gbc)
model_bagged_dt = ensemble_model(ada)


# In[51]:


# Previsão do modelo Light Gradient Boosting Machine
model_pred_lr = predict_model(lightgbm)
model_pred_lr.head()


# In[52]:


# Previsão do modelo Gradient Boosting Classifier
model_pred_dt = predict_model(gbc)
model_pred_dt.head()


# In[53]:


# Previsão do modelo Ada Boost Classifier 
model_pred_rt = predict_model(ada)
model_pred_rt.head()


# # Métricas do modelo

# In[54]:


# Evaluate modelo Light Gradient Boosting Machine
evaluate_model(lightgbm)


# In[55]:


# Evaluate modelo Gradient Boosting Classifier
evaluate_model(gbc)


# In[56]:


# Evaluate modelo Ada Boost Classifier
evaluate_model(ada)


# # Curva roc do modelos

# In[57]:


# Curva roc do modelo Light Gradient Boosting Machine
plot_model(lightgbm)


# In[58]:


# Curva roc do modelo Gradient Boosting Classifier 
plot_model(gbc)


# In[59]:


# Curva roc do modelo Ada Boost Classifier
plot_model(ada)


# # Confusion matrix

# In[60]:


# plot lightgbm - confusion matrix
plot_model(lightgbm, plot = 'confusion_matrix')


# In[61]:


# plot gbc - confusion matrix
plot_model(gbc, plot = 'confusion_matrix')


# In[62]:


# plot ada - confusion matrix
plot_model(ada, plot = 'confusion_matrix')


# # Class report dos modelos

# In[63]:


plot_model(lightgbm, plot = 'class_report')


# In[64]:


plot_model(gbc, plot = 'class_report')


# In[65]:


plot_model(ada, plot = 'class_report')


# # Salvando modelo

# In[67]:


save_model(lightgbm, model_name='best-model')
save_model(gbc, model_name='best-mode2')
save_model(ada, model_name='best-mode3')


# # Conclusão
# 
# Nesse modelo de classificação utlizei 3 modelos machine learning são eles regressão logística, random forest, decision tree o melhores modelos foi o regressão logística, random forest, extra trees, K-NN, ridge classifier esse objetivo do modelo e classificar tumores de câncer de mama são classificado por benignos, malignos. O resultado da matriz de confusão do primeiro modelo de regressão logística teve 132 para maligno e 73 para benigno. O modelo teve resultado ótimos no recal acima de 95,9% teve acerto total ou seja o modelo aprendeu muito
# 
# Referência
# - https://pycaret.org/
# 
# - https://github.com/pycaret/pycaret
# 
# - https://pycaret.readthedocs.io/en/latest/
# 
# - https://medium.com/ensina-ai/pycaret-a-biblioteca-de-aprendizagem-de-m%C3%A1quinas-para-quem-tem-prazo-1c5b09667763

# In[ ]:




