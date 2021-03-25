# -*- coding: utf-8 -*-

# Ejemplo 03. Caso práctico: Regresión polinomica para el caso de la
# productividad marginal del trabajo de una empresa n dedicada al ramo
# de la producción de componentes electrónicos de telematica

# 1. Carguemos las librerías necesarios

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

os.chdir("C:/Users/Pelu/OneDrive - UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO/cosas de Jaff Koalita/Bedu/ML/Sesion 04/Ejercicio03")


# 2. Carguemos los datos necesarios y exploremoslos

data = pd.read_csv("polynomial-regression.csv")
print(data.info())
print(data.head())
print(data.describe())

# 3. Hagamos un breve data wrangling y grafiquemos un scatterplot

x = data.numero_de_trabajadores.values.reshape(-1,1)
y = data.productividad_marginal.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("trabajadores")
plt.ylabel("productividad marginal del trabajo")
plt.show()

# Hagamos una regresión polinomial, con una recta de regresión estimada como
# y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

polynominal_regression = PolynomialFeatures(degree=4)
x_polynomial = polynominal_regression.fit_transform(x,y)

# Hagamos el fit de los datos
linear_regression = LinearRegression()
linear_regression.fit(x_polynomial,y)

# Veamos como se ajusta la nueva regresión polinomica
y_head2 = linear_regression.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.scatter(x,y)
plt.xlabel("trabajadores")
plt.ylabel("productividad marginal del trabajo")
plt.show()

# Calculemos el R^2

from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y,y_head2))

