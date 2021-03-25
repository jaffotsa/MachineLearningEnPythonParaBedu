# -*- coding: utf-8 -*-

# Reto 02.

# A continuación usted va a crear un modelo de regresión lineal múltiple donde el Salario
# de unos trabajadoresestadísticos financieros n depende de su edad y las horas extra
# que trabajan para su empresa.
# Para esto, será necesario que cargue las siguientes librerías

import pandas as pandita
import numpy as algebralineal
from sklearn.linear_model import LinearRegression as regresiones

print("Current Working Directory " , os.getcwd())
os.chdir("C:/Users/Pelu/OneDrive - UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO/cosas de Jaff Koalita/Bedu/ML/Sesion 04/Reto02")

# 2. cargue el csv proporcionado para la clase
# del día de hoy y explore brevemente los datos

data = pd.read_csv("multiple-linear-regression-dataset.csv")
print(data.info())
print(data.head())

# Para este ejercicio use todos los datos disponibles como training dataset. Definirá como y 
# a la variable con Salario, y además, tendra una segunda variable igual a x 
# como un array (no por separado) con las dos variables independientes. 
# PISTA: haga reshape(-1,1 a Salario, su variable y), y use un .iloc de pandas para generar un
# índice para la variable x

x = data.iloc[:,[0,2]].values
y = data.Salario.values.reshape(-1,1)

x

y


# Use regresiones para desarrollar su modelo de regresión múltiple,
# y haga fit en el las dos variables que creo en el paso anterior

multiple_linear_regression = regresiones()
multiple_linear_regression.fit(x,y)

print("b0: ",multiple_linear_regression.intercept_)
print("b1: ", multiple_linear_regression.coef_)

# Genere una predicción de Salario cuando:

# Primero: una persona tiene 10 horas extras de trabajo y tiene 35 años
# Segundo: una persona tiene 5 horas de trabajo y cuenta con 20 años

x_ = np.array([[10,35],[5,20]])
multiple_linear_regression.predict(x_)


# BONUS: Calcule la R2 cuadrada para este modelo de regresión múltiple

y_head = multiple_linear_regression.predict(x) 
from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y,y_head))