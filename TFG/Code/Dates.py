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

features = features.drop(['velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones'], axis=1)

FlowsALDT = features["FlowsALDT"]
i = 0
time = []
x = np.arange(0, 365, 1)
y = np.zeros(365, np.int)

for flight in FlowsALDT:

    time.append(datetime.strptime(FlowsALDT[i][4:].replace(' CET', '').replace(' CEST', ''),'%b %d %H:%M:%S %Y'))
    y[int(datetime.strftime(time[i], '%j'))-1] += 1
    i = i + 1;


# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)


# the scatter plot:
axScatter.scatter(x, y)
plt.xlabel("Dia del año")
plt.ylabel("Numero de aeronaves")

# now determine nice limits by hand:
binwidth = 0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((-lim, lim))
axScatter.set_ylim((-lim, lim))

bins = np.arange(-lim, lim + binwidth, binwidth)

plt.show()

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
