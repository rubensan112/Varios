# Pandas is used for data manipulation
import pandas as pd
import sklearn
import csv




'Importar Datos'
with open('datosTFG.csv', 'r') as csvfile:
    features = pd.read_csv(csvfile,delimiter = ';' )


'Clean Data'

'FL corregir con QNH'
'FlowsALDT pasar a un formato de fecha?'

print("Finish Ejecution")










