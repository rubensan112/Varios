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



#Load Data, and create DataFrame
features = pd.read_pickle("dataTimes.pkl")
features_old = pd.read_csv('datosTFG.csv', delimiter=';')

features_categories = features["RECAT-EU"]

y = {"A":0,"B":0,"C":0,"D":0,"E":0,"F":0}

for category in features_categories:
    if(category == "A"):
        y["A"] += 1
    if (category == "B"):
        y["B"] += 1
    if (category == "C"):
        y["C"] += 1
    if (category == "D"):
        y["D"] += 1
    if (category == "E"):
        y["E"] += 1
    if (category == "F"):
        y["F"] += 1


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
labels = ('A', 'B', 'C', 'D', 'E', 'F')
y_pos = np.arange(len(labels))
performance = [y[element] for element in y]
error = np.random.rand(len(labels))

ax.barh(y_pos, performance, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Numero de aeronaves')
ax.set_ylabel('Categoria RECAT-EU')

plt.show()


features_ROT = features["MultiROT"]

categoryA = []
categoryB = []
categoryC = []
categoryD = []
categoryE = []
categoryF = []


i = 0
for item in features_categories:
    if(item == "A"):
        categoryA.append(features_ROT[i])
    if (item == "B"):
        categoryB.append(features_ROT[i])
    if (item == "C"):
        categoryC.append(features_ROT[i])
    if (item == "D"):
        categoryD.append(features_ROT[i])
    if (item == "E"):
        categoryE.append(features_ROT[i])
    if (item == "F"):
        categoryF.append(features_ROT[i])
    i += 1

medianA = statistics.median(categoryA)
medianB = statistics.median(categoryB)
medianC = statistics.median(categoryC)
medianD = statistics.median(categoryD)
medianE = statistics.median(categoryE)
medianF = statistics.median(categoryF)

print('Media A {}'.format(statistics.median(categoryA)))
print('Media B {}'.format(statistics.median(categoryB)))
print('Media C {}'.format(statistics.median(categoryC)))
print('Media D {}'.format(statistics.median(categoryD)))
print('Media E {}'.format(statistics.median(categoryE)))
print('Media F {}'.format(statistics.median(categoryF)))


i = 0
baseline_pred = []
for item in features_categories:
    if (item == "A"):
        baseline_pred.append(medianA)
    if (item == "B"):
        baseline_pred.append(medianB)
    if (item == "C"):
        baseline_pred.append(medianC)
    if (item == "D"):
        baseline_pred.append(medianD)
    if (item == "E"):
        baseline_pred.append(medianE)
    if (item == "F"):
        baseline_pred.append(medianF)
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

for i in range(int(max(baseline_errors))):
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

print("Hello")