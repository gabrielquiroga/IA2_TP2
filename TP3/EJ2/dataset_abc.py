import pandas as pd
import numpy as np


def genera_data():
    datos = pd.read_csv('dataset_reducido.csv', header=0, na_filter=False)

    labels = np.zeros(datos.shape[0])
    dataset = np.zeros([datos.shape[0], datos.shape[1] - 1])
    for i in range(0, datos.shape[0]):
        labels[i] = datos.iloc[i][0]
        dataset[i][:] = datos.iloc[i][1:]
    for i in range(0, len(dataset)):
        for j in range(0, len (dataset[0])):
            dataset[i][j] = float(dataset[i][j]/255.0)
    return labels, dataset

    
def dataset_size():
    datos = pd.read_csv('dataset_reducido.csv', na_filter=False)
    cant_datos = datos.shape[0]  # shape[0] para filas, shape[1] para columnas
    return cant_datos

def genera_data_inicializacion():
    datos = pd.read_csv('dataset_inicializacion.csv', header=0, na_filter=False)

    labels = np.zeros(datos.shape[0])
    dataset = np.zeros([datos.shape[0], datos.shape[1] - 1])
    for i in range(0, datos.shape[0]):
        labels[i] = datos.iloc[i][0]
        dataset[i][:] = datos.iloc[i][1:]
    for i in range(0, len(dataset)):
        for j in range(0, len (dataset[0])):
            dataset[i][j] = float(dataset[i][j]/255.0)

    return labels, dataset
