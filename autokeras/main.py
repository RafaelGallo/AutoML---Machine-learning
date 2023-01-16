# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Biblioteca AutoML autokeras
import autokeras as ak

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

# Base geral
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

x = data["stroke"]

# Biblioteca AutoML autokeras
import autokeras as ak

# Initialize the structured data classifier.
# It tries 3 different models.
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)  

# Feed the structured data classifier with training data.
clf.fit(data, x, epochs=10)

# Predict with the best model.
predicted_y = clf.predict(data)
predicted_y

# Evaluate the best model with testing data.
print(clf.evaluate(data, x))

# Modelo AutoKeras - 2 - Data Format
x_train = pd.read_csv("healthcare-dataset-stroke-data.csv")
y_train = x_train.pop("stroke")

# You can also use pandas.DataFrame for y_train.
y_train = pd.DataFrame(y_train)
print(type(y_train))  # pandas.DataFrame

# You can also use numpy.ndarray for x_train and y_train.
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

# Preparing testing data.
x_test = pd.read_csv("healthcare-dataset-stroke-data.csv")
y_test = x_test.pop("stroke")

# It tries 10 different models.
clf = ak.StructuredDataClassifier(overwrite=True, 
                                  max_trials=3)

# Feed the structured data classifier with training data.
clf.fit(x_train, 
        y_train, 
        epochs=10)

# Predict with the best model.
predicted_y = clf.predict(x_test)
predicted_y

# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

import tensorflow as tf

train_set = tf.data.Dataset.from_tensor_slices((x_train.astype(np.unicode), y_train))
test_set = tf.data.Dataset.from_tensor_slices((x_test.to_numpy().astype(np.unicode), y_test))

clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)

# Feed the tensorflow Dataset to the classifier.
clf.fit(train_set, epochs=10)

# Predict with the best model.
predicted_y = clf.predict(test_set)

# Evaluate the best model with testing data.
print(clf.evaluate(test_set))

# Customized Search Space
input_node = ak.StructuredDataInput()
output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=3
)
clf.fit(x_train, y_train, epochs=10)

input_node = ak.StructuredDataInput()
output_node = ak.CategoricalToNumerical()(input_node)
output_node = ak.DenseBlock()(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
)
clf.fit(x_train, y_train, epochs=1)
clf.predict(x_train)

model = clf.export_model()

model.summary()

print(x_train.dtype)

# numpy array in object (mixed type) is not supported.
# convert it to unicode.

pred_mod = model.predict(x_train.astype(np.unicode))
pred_mod
