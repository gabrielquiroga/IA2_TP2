#
# 12/06/19
#
#


import math
import numpy as np
import matplotlib.pylab as plt
from dataset_abc import genera_data
from dataset_abc import dataset_size
from dataset_abc import genera_data_inicializacion

NEURONAS_ENTRADA = 28*28
MAPA_X = 28
MAPA_Y = 28
MAPA_Z = NEURONAS_ENTRADA
NEURONAS_MAPA = MAPA_X*MAPA_Y
PORC_EJ_CLAS = 0.3
PORC_EJ_TEST = 0.1
EPOCHS = 1

Wijk = np.zeros([MAPA_X, MAPA_Y, MAPA_Z])
Wijk = np.random.rand(MAPA_X, MAPA_Y, MAPA_Z)
Wentrada  = np.zeros(MAPA_Z)
Wmapa = np.zeros(MAPA_Z)
mapa = np. zeros([MAPA_X, MAPA_X])

print('Antes del genera_data...')
labels, dataset = genera_data()
print('Pasó el genera_data...')

CANT_EJ = dataset_size() - 5000
CANT_EJ_TEST = int(CANT_EJ * PORC_EJ_TEST)
CANT_EJ_CLAS = int(CANT_EJ * PORC_EJ_CLAS)
CANT_EJ_TRAINING = CANT_EJ - CANT_EJ_TEST - CANT_EJ_CLAS

h = 1
Ri = 3
Rf = 0
tR = CANT_EJ_TRAINING
def n_radio(mu):
    # Calcula un radio variable encada iteración, hasta un máximo de 6 vecinos. Aunque se puede agrandar
    # hardcodeando, debería hacerlo de alguna forma genérica.

    Radio = Ri + (Rf - Ri)*(mu/tR)
    if Radio >= 5.5:
        Radio = 6
    elif Radio >= 4.5 and Radio < 5.5:
        Radio = 5
    elif Radio >= 3.5 and Radio < 4.5:
        Radio = 4
    elif Radio >= 2.5 and Radio < 3.5:
        Radio = 3
    elif Radio >= 1.5 and Radio < 2.5:
        Radio = 2
    elif Radio >= 0.5 and Radio < 1.5:
        Radio = 1
    else:
        Radio = 0
    return Radio


#ALFA = 10
ALFAi = 2
ALFAf = 0.01
tALFA = CANT_EJ_TRAINING
def n_alfa(mu):
    # Calcula un alfa que varía en cada iteración.

    ALFA = ALFAi*pow(ALFAf/ALFAi, mu/tALFA)
    return ALFA


def calculo_metrica(Wentrada, Wmapa):
    # Calcula la distancia euclideana entre 2 vectores de pesos.

    d_euclideana = 0
    dist = 0
    for k in range(0, MAPA_Z):
        dist += pow(Wentrada[k] - Wmapa[k], 2)
    d_euclideana = pow(dist, 0.5)
    return d_euclideana

def comparacion(Wentrada, Wijk):
    # Obtiene la neurona más próxima a la entrada.

    min_x = 0
    min_y = 0
    distancias = np.zeros([MAPA_X, MAPA_Y])
    for i in range(0, MAPA_X):
        for j in range(0, MAPA_Y):
            Wmapa = Wijk[i][j][:]
            distancias[i][j] = calculo_metrica(Wentrada, Wmapa)
    min_distance = distancias[min_x][min_y]
    for i in range(0, MAPA_X):
        for j in range(0, MAPA_Y):
            if distancias[i][j] <= min_distance:
                min_distance = distancias[i][j]
                min_x = i
                min_y = j
    return min_x, min_y

def aprendizaje(Wentrada, Wijk, ALFA, Radio):
    # Modifica los pesos de la neurona más adecuada a la entrada y los de las neuronas
    # vecinas a ella.

    min_x, min_y = comparacion(Wentrada, Wijk)
    for i in range(min_x - Radio, min_x + Radio + 1):
        for j in range(min_y - Radio, min_y + Radio + 1):

            # Con esta porción de código se logra prolongar las neuronas vecinas al otro lado del mapa
            # en caso de que se encuentren mas allá de los bordes o de las esquinas. Es decir, se logra
            # hacer un mapa equivalente a la superficie de una esfera.
            if i < 0:
                ii = MAPA_X + i 
            elif i >= MAPA_X:
                ii = i - MAPA_X 
            else:
                ii = i
            
            if j < 0:
                jj = MAPA_Y + j 
            elif j >= MAPA_Y:
                jj = j - MAPA_Y 
            else:
                jj = j

            for k in range(0, NEURONAS_ENTRADA):
                    Wijk[ii][jj][k] += ALFA*(Wentrada[k] - Wijk[ii][jj][k])

    return Wijk, min_x, min_y


#MAIN--------------------------------------------------------------------------------
print('Entrenando...')
for e in range(0, EPOCHS):
    for mu in range(0, CANT_EJ_TRAINING):
        ALFA = n_alfa(mu)
        Radio = n_radio(mu)
        Wentrada = dataset[mu][:]
        Wijk, min_x, min_y = aprendizaje(Wentrada, Wijk, ALFA, Radio)
        mapa[min_x, min_y] += 1
        print(mu)

# LVQ-----------------------------------------------------------------------------------
print('\nIniciando LVQ...\n')
valor_na = -5 #Valor que va a indicar que no se ha asignado nada a esa neurona.
labels_map = np.zeros([MAPA_X, MAPA_Y])
for i in range(0, MAPA_X):
    for j in range(0, MAPA_Y):
        labels_map[i][j] = valor_na
tasa_de_aciertos = 0
for mu in range(CANT_EJ_TRAINING, CANT_EJ_TRAINING + CANT_EJ_CLAS):
    ALFA = n_alfa(mu)
    Wentrada = dataset[mu][:]
    clase_patron = labels[mu]
    min_x, min_y = comparacion(Wentrada, Wijk)
    clase_ganadora = labels_map[min_x][min_y]
 
    if clase_ganadora == valor_na:
        labels_map[min_x][min_y] = clase_patron
        # print('Valor asignado.')
    elif clase_ganadora == clase_patron:
        for k in range(0, MAPA_Z):
            Wijk[min_x][min_y][k] = Wijk[min_x][min_y][k] + ALFA*(Wentrada[k] - Wijk[min_x][min_y][k])
        # print('Acierto: ', clase_patron, ' ', clase_ganadora, '<----')
    else:
        for k in range(0, MAPA_Z):
            Wijk[min_x][min_y][k] = Wijk[min_x][min_y][k] - ALFA*(Wentrada[k] - Wijk[min_x][min_y][k])
        # print('Error: ', clase_patron, ' ', clase_ganadora)
    print(mu)

# TEST-----------------------------------------------------------------------------
print('\nEvaluando...\n')
cantidad_de_aciertos = 0
cantidad_de_errores = 0
for mu in range(CANT_EJ_TRAINING + CANT_EJ_CLAS, CANT_EJ_TRAINING + CANT_EJ_CLAS + CANT_EJ_TEST):
    Wentrada = dataset[mu][:]
    clase_patron = labels[mu]
    min_x, min_y = comparacion(Wentrada, Wijk)
    clase_ganadora = labels_map[min_x, min_y]
    if clase_ganadora == clase_patron:
        cantidad_de_aciertos += 1
        print('Acierto: ', clase_patron, ' ', clase_ganadora, '<----')
    else:
        cantidad_de_errores += 1
        print('Error: ', clase_patron, ' ', clase_ganadora)

print(labels_map)
print('Cantidad de aciertos: ', cantidad_de_aciertos)
print('Ejemplos de test: ', CANT_EJ_TEST)
print(cantidad_de_aciertos/CANT_EJ_TEST, '%')

# Plotea el mapa con los valores de 0 a 25 que representan acada una de las letras. Además también tiene el
# el valor -5 que representa una neurona a la que no sele asignó ningún valor.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
plt.imshow(labels_map, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()
