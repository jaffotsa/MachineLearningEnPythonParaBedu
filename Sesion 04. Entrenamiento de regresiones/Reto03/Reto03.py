# -*- coding: utf-8 -*-

# Reto 03. Ante usted tiene datos de salario y posición laboral, donde los que se percibe
# depende de la posición laboral. Primero, deberá hacer uso de una regresión
# lineal, y en un segundo paso, hacer uso de una no lineal para poder explicar las ventajas de pensar más allá en términos
# de una no relación lineal ¡Éxito!

# Carguemos las librerías necesarias

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Current Working Directory " , os.getcwd())
os.chdir("C:/Users/Pelu/OneDrive - UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO/cosas de Jaff Koalita/Bedu/ML/Sesion 04/Reto03")


# 1. Importe el dataset. Para este ejercicio no deberá partir entre dataset de entrenamiento
# y prueba


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# 2. Haga una regresión lineal con los datos x e y ajustados.

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 2.5 Visualice su regresión lineal

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# 3. Desarrolle su modelo de regreisón polinomial (nota: use from sklearn.preprocessing import PolynomialFeatures
# despues use PolynomialFeatures con el argumento de degree experimentando distintos grados
#, para ello puede ir graficando su regreisón estimada, hasta llegar a la que haga
# mejor ajuste visualmente. Tambien use su_regresion.fit_transform('su variable x'))

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# 4. Visualice su regresión no lineal

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# 5. Genere una predicción con la regresion lineal para la posición laboral 12 con 
# lin_reg.predict, y ahora genere una con la no lineal (CONSEJO SALVA VIDAS: 
# se hace con la misma base de la regreison lineal pero agregue .suRegresionNoLineal.fit_transform)

lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# 6. BONUS: Visualice su regresión no lineal con una suavización de la recta

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()