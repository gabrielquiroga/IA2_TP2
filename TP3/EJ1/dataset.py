import pandas as pd
import numpy as np

numfilas=890
datos = pd.read_csv('titanic_train.csv', header=0, na_filter=False)

datosfiltrados=[datos['Survived'],datos['Pclass'],datos['Sex'],datos['Age'],datos['Embarked']]
datostranspuestos = np.transpose(datosfiltrados)
print(datostranspuestos)
if datostranspuestos[19][3]== '':
    print('esto es nan')


datostranspuestosfiltrados = np.array([v for v in datostranspuestos if v[3] != ''])

print(datostranspuestosfiltrados)

print (datostranspuestosfiltrados[1][2])

df = pd.DataFrame(datostranspuestosfiltrados)
df.to_csv('datos_filtrados.csv')
