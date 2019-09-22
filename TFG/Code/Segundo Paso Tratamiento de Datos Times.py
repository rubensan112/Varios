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
features = pd.read_pickle("dataRECAT-EU.pkl")
features_old = pd.read_csv('datosTFG.csv', delimiter=';')

features = features.reset_index()
FlowsALDT = features["FlowsALDT"]

y = np.zeros(365, np.int)
time = []
week_number = []
hour = []
i = 0
for flight in FlowsALDT:
    time.append(datetime.strptime(FlowsALDT[i][4:].replace(' CET', '').replace(' CEST', ''),'%b %d %H:%M:%S %Y'))
    week_number.append(float(datetime.strftime(time[i], '%W')))
    hour.append(float(datetime.strftime(time[i], '%H')))
    i = i + 1;


first_date = time[0]
last_date = time[len(time)-1]

features["datetimes"] = time
features["week_number"] = week_number
features["hour"] = hour


x = np.arange(0, 24, 1)
y = np.zeros(24, np.int)
rot_m = np.zeros(24, np.int)
rot_m_e = np.zeros(24, np.int)
i = 0
for hour_flight in features["hour"]:
    y[int(hour_flight)] += 1
    rot_m[int(hour_flight)] += features['MultiROT'][i]
    rot_m_e[int(hour_flight)] += 1
    i += 1

medium_rot = rot_m / rot_m_e

def gbar(ax, x, y, width=0.5, bottom=0):
    X = [[.6, .6], [.7, .7]]
    for left, top in zip(x, y):
        left = left - width/2
        right = left + width
        ax.imshow(X, interpolation='bicubic', cmap=plt.cm.copper,
                  extent=(left, right, bottom, top), alpha=1)




xmin, xmax = xlim = min(x)-1, max(x)+1
ymin, ymax = ylim = min(y), max(y)

fig, ax = plt.subplots()
ax.set(xlim=xlim, ylim=ylim, autoscale_on=False)

plt.xlabel("Hora")
plt.ylabel("Numero de casos")

gbar(ax, x, y, width=0.7)
ax.set_aspect('auto')
plt.show()







features.to_pickle("dataTimes.pkl")

print("Hello")