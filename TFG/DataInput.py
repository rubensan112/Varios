import csv
import pandas


'Importar Datos'
features = []
with open('datosTFG.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, dialect = 'excel', delimiter = ';')
    for row in csvreader:
        features.append(row)


print("Importados {} casos".format(len(features)-1))

'Data cleaning'












'Paso 1: Obtener Velocidades de Referencia'
'''
Para ello necesito la primera velocidad IAS recibida en el radar de Modo-S, a una altitud inferior a 1000ft, cuando
el viento es de cara y con una intensidad de 0 a 10 nudos.

'''

'''
Con estas velocidades se calcula los tiempos para recorrer las 
separaciones minimias establecidas en RECAT-EU.

Despues se compara con el ROT de la aeronave precedente.

Ademas  solo se contemplan cuando el viento es de cara y de 0 a 10 nudos.
'''