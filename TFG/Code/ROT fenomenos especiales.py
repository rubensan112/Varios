import pandas as pd
import webbrowser
import copy
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
#import seaborn as sns
from pandas.plotting import table
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter




#Load Data, and create DataFrame
features = pd.read_csv('datosTFG.csv', delimiter=';')
print('The shape of our features is:', features.shape)

#TRansform this var in float
features["air_temperature_celsiu"] = pd.to_numeric(features["air_temperature_celsiu"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_station"] = pd.to_numeric(features["presion_atmosferica_level_station"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_sea"] = pd.to_numeric(features["presion_atmosferica_level_sea"].str.replace(',', '.').astype(float))
features["tempeartura_condensacion"] = pd.to_numeric(features["tempeartura_condensacion"].str.replace(',', '.').astype(float))

features_rafaga = features[["velocidad_rafaga","MultiROT"]]
print('Numero de datos rafaga: {}'.format(len(features_rafaga)))
features_rafaga = features_rafaga.dropna(axis=0)
print('Numero de datos rafaga: {}'.format(len(features_rafaga)))
one = features_rafaga.describe()


features_fem = features[["fenomeno_especial", "MultiROT"]]
print('Numero de datos fem: {}'.format(len(features_fem)))
features_fem = features_fem.dropna(axis=0)
print('Numero de datos fem: {}'.format(len(features_fem)))
two = features_fem.describe()

features_fem_op = features[["fenomeno_especial_operaciones", "MultiROT"]]
print('Numero de datos fem op: {}'.format(len(features_fem_op)))
features_fem_op = features_fem_op.dropna(axis=0)
print('Numero de datos fem: {}'.format(len(features_fem_op)))
three = features_fem_op.describe()

four = features.describe()

features_whf = features.drop(['velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones'], axis=1)

'''
print('Lenght data before clean: {}'.format(len(features)))

if(features.isnull().values.any()):
    empty_values = features.isnull().sum().sum()
    print('Datos vacios: {}'.format(empty_values))

    nan_rows = features[features.isnull().any(1)]
    nan_rows_list = features[features.isnull().any(1)].index
    nan_rows_num = len(nan_rows_list)

print('Rows con datos Nan: {}'.format(nan_rows_num))

features = features.dropna(axis=0)
'''


#corr_table = pd.DataFrame.corr(features, method='pearson')



f = open("test.html","w")

f.write(one.to_html())

f.write(two.to_html())

f.write(three.to_html())

f.write(four.to_html())

f.close()

webbrowser.open('test.html')