import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import time


features_new = pd.read_pickle("../data_treatment/data/diff_angle_runway_wind_float.pkl")

features = pd.DataFrame()
features["MultiROT"] = features_new["MultiROT"]
#features["MultiRunway"] = features_new["MultiSalidaRapida"]
features["RECAT"] = features_new["RECAT"]
features["diff_angle_runway_wind"] = features_new["diff_angle_runway_wind"]
features["velocidad_viento"] = features_new["velocidad_viento"]

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

#Transformar en unitario
featuresx = features.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(featuresx)
features = pd.DataFrame(x_scaled)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state = 42)

baseline_preds = test_features[:, feature_list.index('RECAT_F')]
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

print("Finish Execution")