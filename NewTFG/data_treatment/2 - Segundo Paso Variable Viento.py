import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt # Linea importante

df = pd.read_pickle("../data_treatment/data/firstRECAT-EU.pkl")

direccion_viento_new = df["direccion_viento"]
runway = df["MultiRunway"]

dict_runway = {
    "02": 200,
    "25R": 70,
    "25L": 70,
    "07R": 250,
    "07L": 250
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
    diff_angle_runway_wind.append(abs(runway_angle - angle_wind[i]))
    diff_angle_runway_wind_full.append(runway_angle - angle_wind[i])
    i+=1


df["diff_angle_runway_wind"] = diff_angle_runway_wind
df["diff_angle_runway_wind_full"] = diff_angle_runway_wind_full

### Variable Viento ###

print("-----------------------------------------------------------------")
print(rp.summary_cont(df['MultiROT'].groupby(df['diff_angle_runway_wind'])).to_string())
print(rp.summary_cont(df['MultiROT'].groupby(df['diff_angle_runway_wind_full'])).to_string())


# Opcion 1, guardar como float
df.to_pickle("data/diff_angle_runway_wind_float.pkl")

# Opcion 2, guardar como string
df.to_pickle("data/diff_angle_runway_wind_string.pkl")




print("-----------------------------------------------------------------")
results = ols('MultiROT ~ C(diff_angle_runway_wind)', data=df).fit()
print("Estudy of terminal with Anova Method")
print(results.summary())
print("-----------------------------------------------------------------")
print(sm.stats.anova_lm(results, typ=2))