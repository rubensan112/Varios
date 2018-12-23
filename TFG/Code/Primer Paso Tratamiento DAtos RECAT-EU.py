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
import statistics
import json

import os

#Load Data, and create DataFrame
features = pd.read_csv('datosTFG.csv', delimiter=';')
print('The shape of our features is:', features.shape)

#TRansform this var in float
features["air_temperature_celsiu"] = pd.to_numeric(features["air_temperature_celsiu"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_station"] = pd.to_numeric(features["presion_atmosferica_level_station"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_sea"] = pd.to_numeric(features["presion_atmosferica_level_sea"].str.replace(',', '.').astype(float))
features["tempeartura_condensacion"] = pd.to_numeric(features["tempeartura_condensacion"].str.replace(',', '.').astype(float))

features_aircraft = features["FlowsAircraft"]
features_runway = features["MultiRunway"]

list_runway = []
list_aircraft = []
list_lost_aircraft = []

for item in features_aircraft:
    if not(item in list_aircraft):
        list_aircraft.append(item)

for item in features_runway:
    if not(item in list_runway):
        list_runway.append(item)

f = open("RECAT-EU Category.json","r")

json_content = json.loads(f.read())

f.close()

features['RECAT-EU'] = 0

i = 0

for item in list_aircraft:
    try:
        test = json_content[item]
    except:
        if not (item in list_lost_aircraft):
            list_lost_aircraft.append(item)
    i += 1

i=0
index_list = []
''''''
for item in features_aircraft:
    if(item in list_lost_aircraft):
        #Acceder al datafreme dentro de un loop es muy ineficiente.
        index_list.append(i)
    i += 1
features = features.drop(index_list)
print("tamaño de la muestra: {}" .format(len(features)))

index_list = []
cat_list = []
i = 0
for item in features_aircraft:
    try:
        #features["RECAT-EU"][i] = json_content[item]
        cat_list.append(json_content[item])
    except:
        print("La aeronave {} no esta en el dict".format(item))
    i+=1

features["RECAT-EU"] = cat_list


features.to_pickle("dataRECAT-EU.pkl")


if (os.path.exists("dataRECAT-EU.csv")):
    os.remove("dataRECAT-EU.csv")

f = open("dataRECAT-EU.csv","w")

f.write(features.to_csv(index_label = False))

f.close()


print("hello")


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
