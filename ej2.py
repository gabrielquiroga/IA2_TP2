import math
import numpy as np

pos=0
vel=0
ac=0
dt=0.01
t=0
#definicion de umbrales conjutos borrosos
uNF=-20
uNP=-10
uPP=10
uPF=20

while t<=27:
    #Asignacion de los conjuntos borrosos
    #posicion
    if pos<=uNF:
        Pnf=1
        pos1=0
        pos2=0
    if pos>uNF and pos<=uNP:
        Pnp=abs((pos-uNF)/(uNF-uNP))
        Pnf=1-Pnp
        pos1=0
        pos2=1
    if pos>uNP and pos<=0:
        Pze=abs((pos-uNP)/uNP-0)
        Pnp=1-Pze
        pos1=1
        pos2=2
    if pos>0 and pos<=uPP:
        Ppp=abs(pos/uPP)
        Pze=1-Ppp
        pos1=2
        pos2=3
    if pos>uPP and pos<=uPF:
        Ppf=abs((pos-uPP)/(uPF-uPP))
        Ppp=1-Ppf
        pos1=3
        pos2=4 
    if pos>= uPF:
        Ppf=1
        pos1=4
        pos2=4
    #velocidad
    if vel<=uNF:
        Vnf=1
        vel1=0
        vel2=0
    if vel>uNF and vel<=uNP:
        Vnp=abs((vel-uNF)/(uNF-uNP))
        Vnf=1-Vnp
        vel1=0
        vel2=1       
    if vel>uNP and vel<=0:
        Vze=abs((vel-uNP)/uNP-0)
        Vnp=1-Vze
        vel1=1
        vel2=2        
    if vel>0 and vel<=uPP:
        Vpp=abs(vel/uPP)
        Vze=1-Vpp
        vel1=2
        vel2=3
    if vel>uPP and vel<=uPF:
        Vpf=abs((vel-uPP)/(uPF-uPP))
        Vpp=1-Vpf
        vel1=3
        vel2=4
    if vel>= uPF:
        Vpf=1
        vel1=4
        vel2=4

    #Reglas Borrosas para el conjunto borroso fuerza
    MatrizReglas=[]
    for i in range(5):
        