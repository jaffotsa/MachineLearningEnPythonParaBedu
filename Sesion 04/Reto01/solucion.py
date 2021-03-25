# -*- coding: utf-8 -*-

# Reto 01. Entrena un modelo de regresión lineal con los datos proporcionados. Usted
# debe realizar una proyección con una regresión que defina al precio como una función dependiente
# de los pies cuadrados (el espacio en una vivienda). Es libre de desarrollar EDA para entender
# sus datos antes de realizar el ejercicio.

# 1. Importe las librerías necesarias (Pandas, matplotlib, scikit-learn numpy y os si usa spyder u otra IDE
# que no sea Jupyter)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn

print("Current Working Directory " , os.getcwd())
os.chdir("C:/Users/Pelu/OneDrive - UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO/cosas de Jaff Koalita/Bedu/ML/Sesion 04/Reto01")

%matplotlib inlinepath
np.set_printoptions(threshold=np.nan)


# 2. Haga el data wrangling necesario. Recuerde. DEBE FIJAR COMO VARIABLE INDEPENDIENTE A 
# sqft_living Y COMO DEPENDIENTE A price. Como pista, debe convertir a arrat de numpy 
# sus variables y hacerle un reshape de (-1, 1) a sqft_living

dataset = pd.read_csv("kc_house_data.csv")
space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

x
y

# Separe en conjunto entrenamiento y en conjunto de prueba con Scikit Learn. (Pista:
# usa la vieja confiable de xtrain, xtest, ytrain, ytest = ... con la infalibre sublibrería
# from from sklearn.model_selection import train_test_split )

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)


# Ajuste el modelo de regresión en los datos de entrenamiento con el siguiente template
# from sklearn.linear_model import LinearRegression 
# regressor = LinearRegression()
# regressor.fit('tu eje de las x de training', 'tu eje de las y de training')

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Genera tu predicción con regressor.predict
pred = regressor.predict(xtest)

pred

# Visualiza tus resultados con matplotlib
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

# Haz la prueba con el conjunto de prueba y visualiza
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

