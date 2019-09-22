import pandas as pd

features = pd.read_pickle("../data_treatment/data/firstRECAT-EU.pkl")


# Metodo Anova para las diferentes Variables Categoricas


import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt # Linea importante
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.tools import FigureFactory as FF
#import matplotlib.pyplot as plt # Linea importante

df = pd.read_pickle("../data_treatment/data/3_data_with_times.pkl")

index_list = []
for index, value in enumerate(df['air_temperature_celsiu']):
     if str(value) == 'nan':
        index_list.append(index)

df = df.drop(index_list)
rp.summary_cont(df['air_temperature_celsiu'].groupby(df['week_number'])).to_csv('test6.csv', sep=";", decimal=",")
df.boxplot(column="air_temperature_celsiu", by='week_number', figsize=(12, 8))

#Create a boxplot
df.boxplot(column="MultiROT", by='MultiSalidaRapida', figsize=(12, 8))
df.boxplot(column="MultiROT", by='RECAT', figsize=(12, 8))
df.boxplot(column="MultiROT", by='Ramp', figsize=(12, 8))
df.boxplot(column="MultiROT", by='FlowsFlightType', figsize=(12, 8))

### Day ###ç
rp.summary_cont(df['MultiROT'].groupby(df['air_temperature_celsiu'])).to_csv('test6.csv', sep=";", decimal=",")
df.boxplot(column="MultiROT", by='RECAT', figsize=(12, 8))

# Tukey test
mc = MultiComparison(df['MultiROT'], df['week_number'])
result = mc.tukeyhsd()

str(result).replace('.', ',')

### Hour ###ç
#rp.summary_cont(df['MultiROT'].groupby(df['hour'])).to_csv('test3.csv', sep=";", decimal=",")


print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['hour'])).to_string())
print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(hour)', data=df).fit()
print("Estudy of Hour with Anova Method")
table = sm.stats.anova_lm(results, typ=2)
print(results.summary())
df.boxplot(column="MultiROT", by='hour', figsize=(12, 8))
print("-----------------------------------------------------------------")

# Tukey test
mc = MultiComparison(df['MultiROT'], df['hour'])
result = mc.tukeyhsd()

str(result).replace('.', ',')
print(result)
print(mc.groupsunique)


### Ramp -  MultiSalidaRapida ###

results = ols('MultiROT ~ Ramp + MultiSalidaRapida', data=df).fit()
print("Estudy of FlowsFlightType with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))

### RECAT-EU ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['RECAT'])).to_string())
print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(RECAT)', data=df).fit()
print("Estudy of RECAT-EU with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))

### MultiRunway ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['MultiRunway'])).to_string())
print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(MultiRunway)', data=df).fit()
print("Estudy of MultiRunway with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))

### MultiSalidaRapida ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['MultiSalidaRapida'])).to_string())
print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(MultiSalidaRapida)', data=df).fit()
print("Estudy of MultiSalidaRapida with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))

### Ramp ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['Ramp'])).to_string())
print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(Ramp)', data=df).fit()
print("Estudy of Ramp with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))

### terminal ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['terminal'])).to_string())
print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(terminal)', data=df).fit()
print("Estudy of terminal with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))

### FlowsFlightType ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['FlowsFlightType'])).to_string())
print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(FlowsFlightType)', data=df).fit()
print("Estudy of FlowsFlightType with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))



r1 = rp.summary_cont(df['MultiROT'])
r2 = rp.summary_cont(df['MultiROT'].groupby(df['Ramp']))

print(rp.summary_cont(df['MultiROT'].groupby(df['direccion_viento'])).to_string())
print(rp.summary_cont(df['MultiROT'].groupby(df['direccion_viento'])).to_string())


print(rp.summary_cont(df['MultiROT'].groupby(df['terminal'])))
print(rp.summary_cont(df['MultiROT'].groupby(df['direccion_viento'])).set_option('display.max_colwidth', -1))

results = ols('MultiROT ~ C(direccion_viento)', data=df).fit()
print("Estudy of terminal with Anova Method")
print(results.summary())
print(sm.stats.anova_lm(results, typ=2))

results = ols('MultiROT ~ C(Ramp)', data=df).fit()
print("Estudy of terminal with Anova Method")
print(results.summary())
print(sm.stats.anova_lm(results, typ=2))

small_df = df.head(1000)

# Ramp
results = ols('MultiROT ~ C(terminal)', data=small_df).fit()
print("Estudy of terminal with Anova Method")
print(results.summary())
print(sm.stats.anova_lm(results, typ=2))

df = df
new_df = pd.DataFrame(data={})

new_df["MultiROT"] = df["MultiROT"]
new_df["RECAT"] = df["RECAT"]

results = stats.f_oneway(new_df["MultiROT"][new_df['RECAT'] == 'B'],
               new_df["MultiROT"][new_df['RECAT'] == 'C'],
               new_df["MultiROT"][new_df['RECAT'] == 'D'],
               new_df["MultiROT"][new_df['RECAT'] == 'E'],
               new_df["MultiROT"][new_df['RECAT'] == 'F'])

# Ramp
results = ols('MultiROT ~ C(Ramp)', data=df).fit()
print("Estudy of Ramp with Anova Method")
print(results.summary())
print(sm.stats.anova_lm(results, typ=2))

# RECAT-EU
results = ols('MultiROT ~ C(RECAT)', data=df).fit()
print("Estudy of RECAT-EU with Anova Method")
print(results.summary())
print(sm.stats.anova_lm(results, typ=2))










### Example ###

# Loading data
df = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/difficile.csv")
df.drop('person', axis=1, inplace=True)

# Recoding value from numeric to string
df['dose'].replace({1: 'placebo', 2: 'low', 3: 'high'}, inplace=True)

# Gettin summary statistics
#rp.summary_cont(df['libido'])

#rp.summary_cont(df['libido'].groupby(df['dose']))

# ANOVA with scipy.stats
# stats.f_oneway(data_group1, data_group2, data_group3, data_groupN)
stats.f_oneway(df['libido'][df['dose'] == 'high'],
             df['libido'][df['dose'] == 'low'],
             df['libido'][df['dose'] == 'placebo'])


results = ols('libido ~ C(dose)', data=df).fit()
results.summary()

print("Finish Execution")