import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt # Linea importante

# generate related variables
from numpy import mean, array
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
from numpy import cov
from scipy.stats import pearsonr, chisquare
from scipy.stats import spearmanr


df = pd.read_pickle("../data_treatment/data/3_data_with_times.pkl")

### Temperatura ###

### humedad_relativa ###

data1 = array(df["MultiROT"])
data2 = array(df["air_temperature_celsiu"])

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

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)

##########












### humedad_relativa ###

data1 = df["MultiROT"]
data2 = df["humedad_relativa"]

# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1, data2)
pyplot.show()

### Covariance ###

covariance = cov(data1, data2)
print(covariance)

### Pearson’s Correlation ###

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)


### presion_atmosferica_level_sea ###


data1 = df["MultiROT"]
data2 = df["presion_atmosferica_level_sea"]

# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1, data2)
pyplot.show()

### Covariance ###

covariance = cov(data1, data2)
print(covariance)

### Pearson’s Correlation ###

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)

### air_temperature_celsiu ###

data1 = df["MultiROT"]
data2 = df["air_temperature_celsiu"]

# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1, data2)
pyplot.show()

### Covariance ###

covariance = cov(data1, data2)
print(covariance)

### Pearson’s Correlation ###

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)









# https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)

### Test Dataset ###
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1, data2)
pyplot.show()

### Covariance ###

covariance = cov(data1, data2)
print(covariance)

### Pearson’s Correlation ###

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)



