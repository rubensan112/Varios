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
import math

from matplotlib.ticker import MaxNLocator
from collections import namedtuple

#Load Data, and create DataFrame
features = pd.read_pickle("../dataRunway.pkl")
features = features.drop(['index','Callsign','FlowsADEP','FlowsFlightRule','FlowsFlightType','FlowsWake','FlowsAircraft','FlowsEngines','FlowsALDT','air_temperature_celsiu','presion_atmosferica_level_station','presion_atmosferica_level_sea','humedad_relativa','velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones','nubes','visibilidad_horizontal','standKey','Ramp','terminal','TA','VX','VY','datetimes'], axis=1)


salidas =  ['UB', 'P5', 'P3', 'R6', 'R5', 'R3', 'P6', 'G9', 'G8', 'G7', 'P7', 'G6', 'G5', 'G4', 'R1', 'P1', 'R2', 'P4', 'P2', 'R4']
median =   [56.0, 39.0, 34, 50, 42.0, 36.0, 50.0, 51.0, 43, 38.0, 89.0, 40, 47, 53, 41, 47.0, 29, 26.0, 34.0, 22.0]
dict =  {'UB': [18010, 56.0], 'P5': [20990, 39.0], 'P3': [3497, 34], 'R6': [29515, 50], 'R5': [30104, 42.0], 'R3': [2674, 36.0], 'P6': [11808, 50.0], 'G9': [482, 51.0], 'G8': [3899, 43], 'G7': [2738, 38.0], 'P7': [2, 89.0], 'G6': [139, 40], 'G5': [55, 47], 'G4': [1, 53], 'R1': [12739, 41], 'P1': [6662, 47.0], 'R2': [507, 29], 'P4': [78, 26.0], 'P2': [1510, 34.0], 'R4': [14, 22.0]}
pred = []

i =0
for item in features['MultiSalidaRapida']:
    pred = dict[features['MultiSalidaRapida'][i]][1]
    i = i+1

# Baseline errors, and display average baseline error
baseline_errors = abs(features['MultiROT'] - pred)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))

'''
list_salidas = []
i=0
data = features['MultiSalidaRapida']
dat2 = features['MultiRunway']
rot = features['MultiROT']
for item in data:
    salida = [dat2[i],data[i]]
    if not(salida in list_salidas):
        dict = [dat2[i],data[i]]
        list_salidas.append(dict)
    i= i+1

dict = {}
j=0
for i in list_salidas:
    dict[i[1]] = []
features['PredRot'] = 0

for item in rot:
    for i in list_salidas:
        if(features['MultiSalidaRapida'][j] == i[1]):
            dict[i[1]].append(features['MultiROT'][j])
    j = j +1

median = {}
for i in list_salidas:
    median[i[1]] = [len(dict[i[1]]) , statistics.median(dict[i[1]])]
    print('La media de la salida rapida {} es {}'.format(i[1], median[i[1]]))


i = 0
baseline_pred = []
for item in rot:
    if (item == "J"):
    i += 1






'''



print("Finish execution")