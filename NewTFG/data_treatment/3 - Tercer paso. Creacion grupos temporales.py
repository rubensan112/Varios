import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


#Load Data, and create DataFrame
features = pd.read_pickle("data/diff_angle_runway_wind_float.pkl")
#features_old = pd.read_csv('datosTFG.csv', delimiter=';')

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

groups = {
'0' : 'E',
'1': 'E',
'2': 'J',
'3': 'F',
'4': 'F',
'5': 'D',
'6' : 'A',
'7': 'A',
'8': 'B',
'9': 'B',
'10' : 'B',
'11': 'C',
'12': 'C',
'13': 'K',
'14': 'C',
'15': 'B',
'16' : 'B',
'17': 'C',
'18': 'A',
'19': 'L',
'20' : 'D',
'21': 'G',
'22': 'H',
'23': 'I',

}


hour_categories = []
for idx, val in enumerate(features['hour']):
    hour_categories.append(groups[str(int(val))])

groups_week = {
'0' : 'E',
'1': 'E',
'2': 'G',
'3': 'J',
'4': 'I',
'5': 'K',
'6' : 'F',
'7': 'E',
'8': 'B',
'9': 'E',
'10' : 'E',
'11': 'E',
'12': 'D',
'13': 'D',
'14': 'C',
'15': 'C',
'16' : 'C',
'17': 'C',
'18': 'C',
'19': 'C',
'20' : 'B',
'21': 'B',
'22': 'C',
'23': 'B',
'24': 'B',
'25': 'A',
'26': 'B',
'27': 'B',
'28': 'B',
'29' : 'A',
'30': 'B',
'31': 'B',
'32': 'B',
'33' : 'B',
'34': 'B',
'35': 'B',
'36': 'B',
'37': 'C',
'38': 'B',
'39' : 'C',
'40': 'C',
'41': 'B',
'42': 'D',
'43' : 'C',
'44': 'D',
'45': 'D',
'46': 'D',
'47': 'D',
'48': 'E',
'49' : 'E',
'50': 'D',
'51': 'D',
'52': 'H'
}


week_categories = []
for idx, val in enumerate(features['week_number']):
    week_categories.append(groups_week[str(int(val))])

features['hour_categories'] = hour_categories
features['week_categories'] = week_categories

features.to_pickle("data/3_data_with_times.pkl")

#PLOT

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









print("Hello")