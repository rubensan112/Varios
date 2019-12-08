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


features = pd.read_pickle("data/7_data_with_fenomenos_especiales.pkl")

runway_02 = []
for index, value in enumerate(features['MultiRunway']):
    if str(value) == '02':
        runway_02.append(features['TA'][index])


data1 = features['MultiROT']
data2 = features['TA']
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

print('hello world')