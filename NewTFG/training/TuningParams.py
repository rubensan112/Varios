import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


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

test = SVC()
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

tuned_parameters = [
    {'min_samples_split': [2, 3, 4, 5],
     'n_estimators':[10]}
]

rfc = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [10, 20],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_features, train_labels)

print("Best parameters set found on development set:")
print()
print(CV_rfc.best_params_)
print()
print("Grid scores on development set:")
print()
means = CV_rfc.cv_results_['mean_test_score']
stds = CV_rfc.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, CV_rfc.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = test_labels, CV_rfc.predict(test_features)
print(classification_report(y_true, y_pred))
print()



scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()


    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(train_features, train_labels)


    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_labels, clf.predict(test_features)
    print(classification_report(y_true, y_pred))
    print()


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