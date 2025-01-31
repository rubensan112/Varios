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
import tables
from sklearn import preprocessing
import matplotlib.pyplot as plt
#import seaborn as sns

#Load Data, and create DataFrame
features = pd.read_csv('datosTFG.csv', delimiter=';')
print('The shape of our features is:', features.shape)

#Transform Data
features_complete = copy.deepcopy(features)

#TRansform this var in float
features["air_temperature_celsiu"] = pd.to_numeric(features["air_temperature_celsiu"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_station"] = pd.to_numeric(features["presion_atmosferica_level_station"].str.replace(',', '.').astype(float))
features["presion_atmosferica_level_sea"] = pd.to_numeric(features["presion_atmosferica_level_sea"].str.replace(',', '.').astype(float))
features["tempeartura_condensacion"] = pd.to_numeric(features["tempeartura_condensacion"].str.replace(',', '.').astype(float))

#Drop some Vars
features = features.drop(['Callsign','air_temperature_celsiu','presion_atmosferica_level_station','presion_atmosferica_level_sea','FlowsEngines','humedad_relativa','velocidad_viento','tempeartura_condensacion','FlowsRunway','FlowsFlightRule','MultiRunway','MultiSalidaRapida','direccion_viento', 'FlowsFlightType', 'FlowsAircraft', 'FlowsADEP','FlowsALDT','velocidad_rafaga','fenomeno_especial','fenomeno_especial_operaciones','nubes','visibilidad_horizontal','standKey','Ramp','terminal'], axis=1)

test = features.describe()
corr_table = pd.DataFrame.corr(features, method='pearson')

#TRansform this var in float
#features["air_temperature_celsiu"] = pd.to_numeric(features["air_temperature_celsiu"].str.replace(',', '.').astype(float))
#features["presion_atmosferica_level_station"] = pd.to_numeric(features["presion_atmosferica_level_station"].str.replace(',', '.').astype(float))
#features["presion_atmosferica_level_sea"] = pd.to_numeric(features["presion_atmosferica_level_sea"].str.replace(',', '.').astype(float))
#features["tempeartura_condensacion"] = pd.to_numeric(features["tempeartura_condensacion"].str.replace(',', '.').astype(float))

features = pd.get_dummies(features)


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


featuresx = features.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(featuresx)
features = pd.DataFrame(x_scaled)

# Convert to numpy array
features = np.array(features)






'''
i=0
#Cleaning Data

for var in features:
    for value in var:
        check = (not(isinstance(value, float)) or value > 1000 or value < -1000 or value)
        if check:
            features[var].drop()
            print("stop")
    i+=1
    print(i)
'''


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)



#Establish Baseline

# The baseline predictions are 46
baseline_preds = test_features[:, feature_list.index('GS')]
baseline_preds[baseline_preds != 46] = 46

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))


#Train Model

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

start_time = time.time()
# Train the model on training data
rf.fit(train_features, train_labels);

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



'''
'Graphs'
# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')

'''

'''
#print('{}'.format(features.iloc[1,:]))
#test = features.head(1)


features_one_row = features.head(1)
for index, row in features_one_row.iterrows():
    for item in row.keys():
        print('Item: {} = {} es {}'.format(item,row[item], type(row[item])))


#test = features.describe()

#Resultados
f = open("test.html","w")

f.write(test.to_html())

f.close()

webbrowser.open('test.html')
'''

#Resultados
f = open("test.html","w")

f.write(test.to_html())

f.close()

webbrowser.open('test.html')

f = open("test.png", "w")

f.write(test)


print("finish")