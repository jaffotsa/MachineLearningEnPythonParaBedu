# -*- coding: utf-8 -*-

#Importemos las librerías necesarias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("Current Working Directory " , os.getcwd())

# Cambiar el path del working directory

os.chdir("C:/Users/Pelu/OneDrive - UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO/cosas de Jaff Koalita/Bedu/ML/Sesion 04")

# Leamos con pandas el archivo con la data

%matplotlib inlinepath
data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
data.head()

# Hagamos una descripción básica de los datos

data.describe()

# Grafiquemos

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

# Ahora debemos calcular la función de costo (breve explciación teórica)

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# Agreguemos un vector de unos al conjunto de entrenamiento para usarlo como una solución
# indexada para usar la función de costo y el gradiente

data.insert(0, 'Ones', 1)

# Partamos nuestros datos

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# Analicemos lo que acabamos de hacer

X.head()

y.head()

# Aquí hay maña. La función de costo está esperando que le metas una matriz en formato
# de numpy, así que necesitamos convertir a X e y antes de usarlos, y tambien,
# debemos inicializar el "contador" de theta.

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

theta

# Veamos el órden de nuestras matrices:

X.shape

theta.shape

y.shape

# Vamos a computar las función de costo para la solución inicial (con theta = 0)

computeCost(X, y, theta)

# Aquí viene lo bueno... Vamos a definir una función que nos ayude a
# desarrollar el método del gradiente en descenso del parametro theta.

# Ahora, con el método del gradiente en descenso lo que buscamos es 
# minimizar la función de costo

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

# Vamos a inicializar las variables con un alpha mínimo, y con un número de iteraciones
# arbtrario

alpha = 0.01
iters = 1000

# Ahora vamos a correr la funci+on de descenso en gradiente para hacer 
# fit de nuestro intercepto theta en el conjunto de entrenamiento

g, cost = gradientDescent(X, y, theta, alpha, iters)
g

# Ahora vamos a computar el costo de error del modelo entrenado con nuestros parametros
# ajustados
computeCost(X, y, g)

# Vamos a usar el modelo lineal para ver la recta muestral de regresión:

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Proyección')
ax.scatter(data.Population, data.Profit, label='Datos de entrenamiento')
ax.legend(loc=2)
ax.set_xlabel('Población')
ax.set_ylabel('Ganancia')
ax.set_title('Proyección vs. Tamaño de la población')

# Ahora podemos usar el metodo del gradiente y que se vea visualmente
# el método del gradiente en descenso, pues el error cuadratico medio aumentará conforme
# vamos avanzando

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. entrenamiento')