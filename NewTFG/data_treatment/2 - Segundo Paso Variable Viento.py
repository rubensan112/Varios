import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from numpy import cov
from scipy.stats import pearsonr, chisquare
from scipy.stats import spearmanr
import math
from numpy import mean, array
import matplotlib.pyplot as plt # Linea importante

df = pd.read_pickle("data/4_data_with_meteo.pkl")

direccion_viento_new = df["direccion_viento"]
runway = df["MultiRunway"]

dict_runway = {
    "02": 199,
    "25R": 65,
    "25L": 65,
    "07R": 245,
    "07L": 245
}
dict_runway2 = {
    "02": 19,
    "25R": 245,
    "25L": 245,
    "07R": 65,
    "07L": 65
}
# Eliminamos
dict_direccion_viento = {
    "Wind blowing from the north": 180,
    "Wind blowing from the north-northeast": 202.5,
    "Wind blowing from the north-east": 225,
    "Wind blowing from the east-northeast": 247.5,
    "Wind blowing from the east": 270,
    "Wind blowing from the east-southeast": 292.5,
    "Wind blowing from the south-east": 315,
    "Wind blowing from the south-southeast": 337.5,
    "Wind blowing from the south": 0,
    "Wind blowing from the south-southwest": 22.5,
    "Wind blowing from the south-west": 45,
    "Wind blowing from the west-southwest": 67.5,
    "Wind blowing from the west": 90,
    "Wind blowing from the west-northwest": 112.5,
    "Wind blowing from the north-west": 135,
    "Wind blowing from the north-northwest": 157.5,
    "Calm, no wind": 0,
    "variable wind direction": 0
}


new_runway = []
for row in runway:
    new_runway.append(dict_runway[row])

angle_wind = []
for row in direccion_viento_new:
    try:
        if str(row) == 'nan':
            angle_wind.append(0)
        else:
            angle_wind.append(dict_direccion_viento[row])
    except Exception as error:
        print("sdfsdf")

i = 0
diff_angle_runway_wind = []
diff_angle_runway_wind_full = []
for runway_angle in new_runway:
    diff = runway_angle - angle_wind[i]
    if diff < 0:
        diff += 360
    if diff > 180:
        diff -= 360
        diff = abs(diff)
    diff_angle_runway_wind.append(diff)
    i+=1

intensidad_proyectada = []
for index, element in enumerate(df['velocidad_viento']):
    intensidad_proyectada.append(element * math.cos(diff_angle_runway_wind[index]))



df["diff_angle_runway_wind"] = diff_angle_runway_wind
df["intensidad_proyectada"] = intensidad_proyectada

data1 = array(df["MultiROT"])
data2 = array(df["velocidad_viento"])

covariance = cov(data1, data2)
print(covariance)

### Pearson’s Correlation ###

corr, pvalue1 = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

### Spearman’s Correlation ###
corr, pvalue2 = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)
### Variable Viento ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['diff_angle_runway_wind'])).to_string())
print(rp.summary_cont(df['MultiROT'].groupby(df['diff_angle_runway_wind_full'])).to_string())



# Opcion 1, guardar como float
df.to_pickle("data/5_data_with_wind.pkl")

# Opcion 2, guardar como string
#df.to_pickle("data/diff_angle_runway_wind_string.pkl")


df.boxplot(column="MultiROT", by='intensidad_proyectada', figsize=(12, 8))

print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(intensidad_proyectada)', data=df).fit()
print("Estudy of terminal with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))