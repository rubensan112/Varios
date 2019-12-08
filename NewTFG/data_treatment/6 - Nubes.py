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

features = pd.read_pickle("data/5_data_with_wind.pkl")


altura_nubes = []
densidad_nubes = []
for index, value in enumerate(features['nubes']):
    try:
        if value == 'No Significant Clouds':
            densidad_nubes.append('No Significant Clouds')
            altura_nubes.append('750')
        else:
            double_cloud = 0
            if re.search('(Few Clouds)', value, re.I | re.S):
                double_cloud += 1
                densidad_nubes.append('Few Clouds')
                altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))
            if re.search('(Scattered clouds)', value, re.I | re.S):
                double_cloud += 1
                if double_cloud > 1:
                    densidad_nubes[index] = 'more_than_one_altitude'
                    altura_nubes[index] = '510'
                else:
                    densidad_nubes.append('Scattered clouds')
                    altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))
            if re.search('(Broken Clouds)', value, re.I | re.S):
                double_cloud += 1
                if double_cloud > 1:
                    densidad_nubes[index] = 'more_than_one_altitude'
                    altura_nubes[index] = '510'
                else:
                    densidad_nubes.append('Broken Clouds')
                    altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))
            if re.search('(Overcast)', value, re.I | re.S):
                double_cloud += 1
                if double_cloud > 1:
                    densidad_nubes[index] = 'more_than_one_altitude'
                    altura_nubes[index] = '510'
                else:
                    densidad_nubes.append('Overcast')
                    altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))
        altura_nubes[index] = int(altura_nubes[index])

    except Exception as error:
        print('hola')

features['altura_nubes'] = altura_nubes
features['densidad_nubes'] = densidad_nubes

features.to_pickle("data/6_data_with_clouds.pkl")

rp.summary_cont(features['MultiROT'].groupby(features['densidad_nubes'])).to_csv('test8.csv', sep=";", decimal=",")
features.boxplot(column="MultiROT", by='densidad_nubes', figsize=(12, 8))

results = ols('MultiROT ~ C(densidad_nubes)', data=features).fit()
print(results.summary())



data1 = features['MultiROT']
data2 = features['densidad_nubes']
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


rp.summary_cont(features['MultiROT'].groupby(features['altura_nubes'])).to_csv('test7.csv', sep=";", decimal=",")
features.boxplot(column="MultiROT", by='altura_nubes', figsize=(12, 8))

results = ols('MultiROT ~ C(densidad_nubes)', data=features).fit()
print(results.summary())


print('hello')


'''

    try:
        if value == 'No Significant Clouds':
            densidad_nubes.append('No Significant Clouds')
            altura_nubes.append('No Significant Clouds')
        else:
            double_cloud = 0
            if re.search('(Few Clouds)', value, re.I | re.S):
                double_cloud += 1
                densidad_nubes.append('Few Clouds')
                altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))
            if re.search('(Scattered clouds)', value, re.I | re.S):
                double_cloud += 1
                if double_cloud > 1:
                    densidad_nubes[index] = 'more_than_one_altitude'
                    altura_nubes[index] = 'more_than_one_altitude'
                else:
                    densidad_nubes.append('Scattered clouds')
                    altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))
            if re.search('(Broken Clouds)', value, re.I | re.S):
                double_cloud += 1
                if double_cloud > 1:
                    densidad_nubes[index] = 'more_than_one_altitude'
                    altura_nubes[index] = 'more_than_one_altitude'
                else:
                    densidad_nubes.append('Broken Clouds')
                    altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))
            if re.search('(Overcast)', value, re.I | re.S):
                double_cloud += 1
                if double_cloud > 1:
                    densidad_nubes[index] = 'more_than_one_altitude'
                    altura_nubes[index] = 'more_than_one_altitude'
                else:
                    densidad_nubes.append('Overcast')
                    altura_nubes.append(str(re.search('\\).+?(\d{1,5})\sm', value, re.I | re.S).groups()[0]))

'''