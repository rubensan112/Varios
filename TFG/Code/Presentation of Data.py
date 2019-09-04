import pandas as pd
import webbrowser
import copy
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import tools needed for visualization
#from sklearn.tree import export_graphviz
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
import graphviz
#import tree
import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydot



PARAMS = {
    "": ""
}

#Load Data, and create DataFrame
#features_new = pd.read_pickle("dataRunway.pkl")
features_new = pd.read_pickle("dataRunway.pkl")
features_old = pd.read_csv('datosTFG.csv', delimiter=';')


features = pd.DataFrame()
features["MultiROT"] = features_new["MultiROT"]
features["MultiRunway"] = features_new["MultiRunway"]
features["RECAT-EU"] = features_new["RECAT-EU"]
#features["VY"] = features_new["VY"]
#features["FL"] = features_new["FL"]
#features["presion_atmosferica_level_station"] = features_new["presion_atmosferica_level_station"]

features = pd.get_dummies(features)

#Drop some Vars
#features = features.drop(['index','Callsign','direccion_viento','MultiSalidaRapida','tempeartura_condensacion','FL','FlowsADEP','FlowsFlightRule','FlowsFlightType','FlowsWake','FlowsAircraft','FlowsEngines','FlowsALDT','FlowsRunway','air_temperature_celsiu','presion_atmosferica_level_station','presion_atmosferica_level_sea','humedad_relativa','velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones','nubes','visibilidad_horizontal','standKey','Ramp','terminal','TA','VX','VY','datetimes', 'velocidad_viento', 'week_number', 'hour'], axis=1)

#features = features.drop(['index','Callsign','direccion_viento','MultiRunway','tempeartura_condensacion','FL','FlowsADEP','FlowsFlightRule','FlowsFlightType','FlowsWake','FlowsAircraft','FlowsEngines','FlowsALDT','FlowsRunway','air_temperature_celsiu','presion_atmosferica_level_station','presion_atmosferica_level_sea','humedad_relativa','velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones','nubes','visibilidad_horizontal','standKey','Ramp','terminal','TA','VX','VY','datetimes', 'velocidad_viento', 'week_number', 'hour'], axis=1)
#features = features.drop(['index','Callsign','direccion_viento','MultiRunway','tempeartura_condensacion','FL','FlowsADEP','FlowsFlightRule','FlowsFlightType','FlowsWake','FlowsAircraft','FlowsEngines','FlowsALDT','FlowsRunway','air_temperature_celsiu','presion_atmosferica_level_station','presion_atmosferica_level_sea','humedad_relativa','velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones','nubes','visibilidad_horizontal','standKey','Ramp','terminal','TA','VX','VY','datetimes', 'week_number', 'hour', 'Diff_heading_runway'], axis=1)
#features = features.drop(['index','Callsign','direccion_viento','MultiRunway','tempeartura_condensacion','FL','FlowsADEP','FlowsFlightRule','FlowsFlightType','FlowsWake','FlowsAircraft','FlowsEngines','FlowsALDT','FlowsRunway','air_temperature_celsiu','presion_atmosferica_level_station','presion_atmosferica_level_sea','humedad_relativa','velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones','nubes','visibilidad_horizontal','standKey','Ramp','terminal','TA','VX','VY','datetimes', 'velocidad_viento', 'week_number', 'hour', 'Diff_heading_runway'], axis=1) # BEST
#features = features.drop(['index','Callsign','direccion_viento','MultiRunway','tempeartura_condensacion','FL','FlowsADEP','FlowsFlightRule','FlowsFlightType','FlowsWake','FlowsAircraft','FlowsEngines','FlowsALDT','FlowsRunway','air_temperature_celsiu','presion_atmosferica_level_station','presion_atmosferica_level_sea','humedad_relativa','velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones','nubes','visibilidad_horizontal','standKey','Ramp','terminal','TA','VX','VY','datetimes', 'velocidad_viento', 'GS', 'week_number', 'hour', 'Diff_heading_runway'], axis=1)
#features = pd.get_dummies(features)

print('Lenght data before clean: {}'.format(len(features)))

if(features.isnull().values.any()):
    empty_values = features.isnull().sum().sum()
    print('Datos vacios: {}'.format(empty_values))

    nan_rows = features[features.isnull().any(1)]
    nan_rows_list = features[features.isnull().any(1)].index
    nan_rows_num = len(nan_rows_list)
    print('Rows con datos Nan: {}'.format(nan_rows_num))


features = features.dropna(axis=0)

print('Lenght data after clean: {}'.format(len(features)))


#Now separete the data into features and targets(label)

# Labels are the values we want to predict
labels = np.array(features['MultiROT'])

# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('MultiROT', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

#Transformar en unitario
featuresx = features.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(featuresx)
features = pd.DataFrame(x_scaled)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state = 42)

medianA = 50
medianB = 51
medianC = 53
medianD = 45
medianE = 44
medianF = 42

#Establish Baseline

# The baseline predictions are 46
baseline_preds = test_features[:, feature_list.index('RECAT-EU_F')]
baseline_preds[baseline_preds != 46] = 46

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))


#Train Model

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=300, random_state=223, min_samples_split=15, min_samples_leaf=15)

start_time = time.time()
# Train the model on training data
rf = rf.fit(train_features, train_labels)

print("--- %s seconds ---" % (time.time() - start_time))

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'seconds')

#Determine Performance Metrics

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

i = 0
for feature_importances in rf.feature_importances_:
    print("feature {0}: has a importance: {1}".format(feature_list[i], feature_importances))
    i += 1

# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot') #El pydot no funka
# Write graph to a png file
graph.write_png('tree.png')




graph = graphviz.Source('final.dot')

file = open('final.png')

graph.render("final2", view=True)
graph.view('final.dot')

graph.save('yokese.gv')
graph.render('tree_new.dot', view=True)

with open('testing_graph_write', 'w+') as file:
    file.write(graph)

# dot -Tpng tree.dot -o tree.png
# Use dot file to create a graph
#(graph, ) = pydot.graph_from_dot_file('tree_new.dot')

# Write graph to a png file
#graph.write_png('tree_new.png')

print("fisnifh")