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
features = pd.read_pickle("dataTimes.pkl")
features_old = pd.read_csv('datosTFG.csv', delimiter=';')

features_runway = features["MultiRunway"]
features_vx = features["VX"]
features_vy = features["VY"]
features_ROT = features["MultiROT"]


list_runway = []

for item in features_runway:
    if not(item in list_runway):
        list_runway.append(item)

i = 0
index_list_02 = []
index_list_25R = []
index_list_25L = []
index_list_07R = []
index_list_07L = []

for item in features_runway:
    if ('02' == item):
        index_list_02.append(i)
    if ('25R' == item):
        index_list_25R.append(i)
    if ('25L' == item):
        index_list_25L.append(i)
    if ('07R' == item):
        index_list_07R.append(i)
    if ('07L' == item):
        index_list_07L.append(i)
    i+=1

data_dict = {'02':{'index':index_list_02},'25R':{'index':index_list_25R},'25L':{'index': index_list_25L},'07R':{'index':index_list_07R},'07L':{'index':index_list_07L}}

data_dict['02']['RumboPista'] = 20
data_dict['25R']['RumboPista'] = 250
data_dict['25L']['RumboPista'] = 250
data_dict['07R']['RumboPista'] = 70
data_dict['07L']['RumboPista'] = 70

features['Diff_heading_runway'] = 0


for runway in data_dict:
    data_dict[runway]['Vx'] = features["VX"][data_dict[runway]['index']]
    data_dict[runway]['Vy'] = features["VY"][data_dict[runway]['index']]
    data_dict[runway]['GS'] = features["GS"][data_dict[runway]['index']]
    data_dict[runway]['TA'] = features["TA"][data_dict[runway]['index']]
    data_dict[runway]['MultiROT'] = features["MultiROT"][data_dict[runway]['index']]
    math_result = data_dict[runway]['Vx'] / data_dict[runway]['GS']
    if(runway == '02' or runway == '07R' or runway == '07L'):
        data_dict[runway]['Heading'] = [(((math.acos(element)*360)/6.28318)-90)*(-1) for element in math_result]
    else:
        data_dict[runway]['Heading'] = [(90-((math.acos(element*(-1)) * 360) / 6.28318) + 180) for element in math_result]
    data_dict[runway]["Diff_heading_runway"] = [abs((element - data_dict[runway]['RumboPista'])) for element in data_dict[runway]['Heading']]
    features['Diff_heading_runway'][data_dict[runway]['index']] = data_dict[runway]["Diff_heading_runway"]


features.to_pickle("dataRunway.pkl")


print("hello")