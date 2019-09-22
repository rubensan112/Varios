import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from numpy import cov
from scipy.stats import pearsonr, chisquare
from scipy.stats import spearmanr


features = pd.read_pickle("data/3_data_with_times.pkl")



# Temperatura
# Eliminar los campos vacios.

i = 0
index_list = []
for value in features['air_temperature_celsiu']:
    if str(value) == 'nan':
        index_list.append(i)
    i += 1

features = features.drop(index_list)
features = features.reset_index()
features = features.drop(['level_0'], axis=1)

i = 0
index_list_1 = []
for value in features['tempeartura_condensacion']:
    if str(value) == 'nan':
        index_list_1.append(i)
    i += 1

features = features.drop(index_list_1)
features = features.reset_index()
features = features.drop(['level_0'], axis=1)


i = 0
index_list_1 = []
for value in features['visibilidad_horizontal']:
    if str(value) == 'nan':
        index_list_1.append(i)
    i += 1

features = features.drop(index_list_1)
features = features.reset_index()
features = features.drop(['level_0'], axis=1)

### Comparacion temperatura y temperatura de condensacion ###

diff_tmp = []
for tmp, tmp_cond in zip(features['air_temperature_celsiu'], features['tempeartura_condensacion']):
    diff_tmp.append(tmp - tmp_cond)

























index_list_test = []
for index, value in enumerate(features['air_temperature_celsiu']):
     if str(value) == 'nan':
        index_list_test.append(index)

index_list_1 = []
for index, value in enumerate(features['direccion_viento']):
     if str(value) == 'nan':
        index_list_1.append(index)

index_list_2 = []
for index, value in enumerate(features['velocidad_viento']):
     if str(value) == 'nan':
        index_list_2.append(index)

index_list_3 = []
for index, value in enumerate(features['nubes']):
     if str(value) == 'nan':
        index_list_3.append(index)

index_list_4 = []
for index, value in enumerate(features['humedad_relativa']):
     if str(value) == 'nan':
        index_list_4.append(index)

index_list_5 = []
for index, value in enumerate(features['visibilidad_horizontal']):
     if str(value) == 'nan':
        index_list_5.append(index)

index_list_6 = []
for index, value in enumerate(features['tempeartura_condensacion']):
     if str(value) == 'nan':
        index_list_6.append(index)

value_list = []
try:
    for value in index_list_6:
        value_list.append(features['tempeartura_condensacion'][value])

except Exception as error:
    print('helo')

features2 = features.drop(index_list_4)

data1 = features["MultiROT"]
data2 = features["air_temperature_celsiu"]

covariance = cov(data1, data2)
print(covariance)

### Pearson’s Correlation ###

corr, p_value = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, p_value1 = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)

hola = features['air_temperature_celsisus' == '5']
