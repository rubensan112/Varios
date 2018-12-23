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




#Load Data, and create DataFrame
features = pd.read_csv('datosTFG.csv', delimiter=';')
print('The shape of our features is:', features.shape)

#TRansform this var in float
features["air_temperature_celsiu"] = pd.to_numeric(features["air_temperature_celsiu"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_station"] = pd.to_numeric(features["presion_atmosferica_level_station"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_sea"] = pd.to_numeric(features["presion_atmosferica_level_sea"].str.replace(',', '.').astype(float))
features["tempeartura_condensacion"] = pd.to_numeric(features["tempeartura_condensacion"].str.replace(',', '.').astype(float))

features_wake = features["FlowsWake"]
features_ROT = features["MultiROT"]
categoryJ = []
categoryH = []
categoryM = []
categoryL = []


i = 0
for item in features_wake:
    if(item == "J"):
        categoryJ.append(features_ROT[i])
    if (item == "H"):
        categoryH.append(features_ROT[i])
    if (item == "M"):
        categoryM.append(features_ROT[i])
    if (item == "L"):
        categoryL.append(features_ROT[i])
    i += 1

medianJ = statistics.median(categoryJ)
medianH = statistics.median(categoryH)
medianM = statistics.median(categoryM)
medianL = statistics.median(categoryL)

print('Media J {}'.format(statistics.median(categoryJ)))
print('Media H {}'.format(statistics.median(categoryH)))
print('Media M {}'.format(statistics.median(categoryM)))
print('Media L {}'.format(statistics.median(categoryL)))




i = 0
baseline_pred = []
for item in features_wake:
    if (item == "J"):
        baseline_pred.append(medianJ)
    if (item == "H"):
        baseline_pred.append(medianH)
    if (item == "M"):
        baseline_pred.append(medianM)
    if (item == "L"):
        baseline_pred.append(medianL)
    i += 1

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_pred - features_ROT)

print("Errores:")
print('Media: ', round(statistics.mean(baseline_errors), 2))
print('Mediana: ', round(statistics.median(baseline_errors), 2))
print('Desviacion Estandar: ', round(statistics.stdev(baseline_errors), 2))
print('Varianza: ', round(statistics.variance(baseline_errors), 2))



y = np.zeros(int(max(baseline_errors)), np.int)
x = np.arange(0, max(baseline_errors), 1)

for i in range(max(baseline_errors)):
    y[i] = len(baseline_errors[baseline_errors == i])


def gbar(ax, x, y, width=0.5, bottom=0):
    X = [[.6, .6], [.7, .7]]
    for left, top in zip(x, y):
        right = left + width
        ax.imshow(X, interpolation='bicubic', cmap=plt.cm.copper,
                  extent=(left, right, bottom, top), alpha=1)



xmin, xmax = xlim = 0, 45
ymin, ymax = ylim = 0, max(y)

fig, ax = plt.subplots()
ax.set(xlim=xlim, ylim=ylim, autoscale_on=False)

plt.xlabel("Error absoluto")
plt.ylabel("Numero de casos")

gbar(ax, x, y, width=0.7)
ax.set_aspect('auto')
plt.show()


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



f = open("test.html","w")

f.close()

webbrowser.open('test.html')