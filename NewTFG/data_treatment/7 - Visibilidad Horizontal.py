
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from numpy import cov
from scipy.stats import pearsonr, chisquare
from scipy.stats import spearmanr

from matplotlib import pyplot
import re
import researchpy as rp
from statsmodels.formula.api import ols

from numpy import mean, array
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
from numpy import cov
from scipy.stats import pearsonr, chisquare
from scipy.stats import spearmanr


features = pd.read_pickle("data/6_data_with_clouds.pkl")


dict = {
    '0,5': 'A',
    '1': 'A',
    '2': 'A',
    '2,5': 'A',
    '3': 'A',
    '3,5': 'A',
    '4': 'A',
    '4,5': 'A',
    '5': 'A',
    '6': 'A',
    '7': 'A',
    '8': 'A',
    '9': 'A',
    '10.0 and more': 'B'

}

new_visibilidad_horizontal = []
for index, value in enumerate(features['visibilidad_horizontal']):
    try:
        new_visibilidad_horizontal.append(dict[value])
    except Exception as error:
        print('hi')

fenomeno_especial = []
countA = 0
countB = 0
countC = 0
rafaga = []
especial = []
especial_operaciones = []
for index, value in enumerate(features['visibilidad_horizontal']):
    try:
        if str(features['velocidad_rafaga'][index]) != 'nan':
            fenomeno_especial.append('A')
            countA += 1
        elif str(features['fenomeno_especial'][index]) != 'nan':
            fenomeno_especial.append('B')
            countA += 1
        elif str(features['fenomeno_especial_operaciones'][index]) != 'nan':
            fenomeno_especial.append('C')
            countA += 1
        else:
            fenomeno_especial.append('D')
    except Exception as error:
        print('hi')

fenomeno_especial_dict = {
    "Drizzle": "Drizzle",
    "Mist": "Drizzle",
    "Light drizzle": "Drizzle",
    "In the vicinity thunderstorm": "A",
    "Light rain": "Nothing",
    "Patches fog": "A",
    "Rain": "Drizzle",
    "Drizzle, mist": "A",
    "Dust/sand whirls (dust devils)": "A",
    "Heavy drizzle": "A",
    "Heavy rain": "A",
    "Heavy thunderstorm, rain": "A",
    "In the vicinity drizzle": "A",
    "In the vicinity fog": "A",
    "In the vicinity rain": "A",
    "In the vicinity shower(s)": "A",
    "In the vicinity shower(s), light drizzle, thunderstorm": "A",
    "In the vicinity shower(s), thunderstorm, light drizzle": "A",
    "In the vicinity thunderstorm, in the vicinity shower(s)": "A",
    "In the vicinity thunderstorm, light drizzle": "A",
    "Light drizzle, mist": "A",
    "Light rain, drizzle": "A",
    "Light rain, drizzle, mist": "A",
    "Light rain, mist": "A",
    "Light rain, thunderstorm": "A",
    "Light shower(s), rain": "A",
    "Light shower(s), rain, in the vicinity thunderstorm": "A",
    "Light thunderstorm, rain": "A",
    "Mist, light drizzle": "A",
    "Mist,partial (covering part of the aerodrome) fog": "A",
    "Mist,patches fog": "A",
    "Rain, in the vicinity thunderstorm": "A",
    "Rain, mist": "A",
    "Shower(s), rain": "A",
    "Shower(s), rain, in the vicinity thunderstorm": "A",
    "Shower(s), rain, thunderstorm": "A",
    "Thunderstorm": "A",
    "Thunderstorm, in the vicinity shower(s)": "A",
    "Thunderstorm, light drizzle": "A",
    "Thunderstorm, rain": "A",

}

group_velocidad_rafaga = []
for index, value in enumerate(features['visibilidad_horizontal']):
    try:
        if str(features['velocidad_rafaga'][index]) != 'nan':
            rafaga.append(features['velocidad_rafaga'][index])
        else:
            rafaga.append(0)
        if str(features['fenomeno_especial'][index]) != 'nan':
            especial.append(fenomeno_especial_dict[features['fenomeno_especial'][index]])
            #especial.append(features['fenomeno_especial'][index])
        else:
            especial.append('Nothing')
        if str(features['fenomeno_especial_operaciones'][index]) != 'nan':
            especial_operaciones.append(features['fenomeno_especial_operaciones'][index])
        else:
            especial_operaciones.append('Nothing')
    except Exception as error:
        print('hi')

features['groups_visibilidad_horizontal'] = new_visibilidad_horizontal
features['groups_velocidad_rafaga'] = rafaga
features['groups_fenomeno_especial'] = especial
features['groups_fenomeno_especial_operaciones'] = especial_operaciones

features.to_pickle("data/7_data_with_fenomenos_especiales.pkl")


rp.summary_cont(features['MultiROT'].groupby(features['fenomeno_especial_operaciones'])).to_csv('test13.csv', sep=";", decimal=",")
features.boxplot(column="MultiROT", by='groups_fenomeno_especial', figsize=(12, 8))
print('Hello world')

data1 = features['MultiROT']
data2 = rafaga
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data2, data1)
pyplot.show()

### Covariance ###

covariance = cov(data1, data2)
print(covariance)

### Pearson’s Correlation ###

corr, pvalue1 = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, pvalue2 = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)
