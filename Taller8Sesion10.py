#"--------------Taller Analitico: Dibujando el Margen--------------"


# Clase A (Círculos): (2,2), (3,3), (4,2)
# Clase B (Equis): (6,6), (7,8), (8,7)

# A(3,3) - B(6,6):  distancia = √(9+9)  = √18  ≈ 4.24   ← el par más cercano
# A(4,2) - B(6,6):  distancia = √(4+16) = √20  ≈ 4.47
# A(2,2) - B(6,6):  distancia = √(16+16)= √32  ≈ 5.66

# La línea óptima es la perpendicular bisectriz entre (3,3) y (6,6): pasa por
# su punto medio (4.5, 4.5) con pendiente -1


# Pregunta 2: ¿Cuáles puntos son los Vectores de Soporte?
# distancia = |x + y - 9| / √2

# Punto	   x+y |x+y-9|	Distancia
# A(2,2)	4	5	3.54
# A(3,3)	6	3	2.12
# A(4,2)	6	3	2.12
# B(6,6)	12	3	2.12
# B(7,8)	15	6	4.24
# B(8,7)	15	6	4.24

# Los Vectores de Soporte son A(3,3), A(4,2) y B(6,6) los tres están exactamente
# a la misma distancia (2.12) de la línea



# Pregunta 3: Si agregamos A(1,1), ¿cambiaría la línea?

# Distancia de (1,1) a la línea = |1+1-9| / √2 = 7 / 1.41 ≈ 4.95

# No, la línea no cambiaría de posición. El nuevo punto (1,1) queda a una
# distancia de 4.95, muchísimo mayor que la distancia de los Vectores de Soporte actuales (2.12)



#"-----------------Taller de Laboratorio: Fronteras No Lineales----------------"

import numpy as np
from sklearn.svm import SVC

# 1. Crear el dataset (X = Coordenadas, Y = Etiquetas binarias 0 o 1)
X = np.array([ [2,2], [3,3], [4,2], [6,6], [7,8], [8,7] ])
Y = np.array([ 0, 0, 0, 1, 1, 1 ])

# 2. Inicializar SVM con Kernel Lineal
modelo_svm = SVC(kernel='linear')

# 3. Entrenar el modelo (Aprender la ecuación del hiperplano)
modelo_svm.fit(X, Y)

# 4. Extraer los Vectores de Soporte descubiertos por la IA
vectores = modelo_svm.support_vectors_
print("Los Vectores de Soporte son:\n", vectores)

# 5. Predicción
nuevo_punto = np.array([[5, 4]])
pred = modelo_svm.predict(nuevo_punto)
print("El punto [5,4] pertenece a la clase:", pred[0])

# Punto 1: Al correr el código del ejemplo, los vectores de soporte impresos 
# coinciden exactamente con los que marcaste con el círculo rojo en tu Taller
# Analítico: (3,3), (4,2) y (6,6).



# Punto 2 y 3: Agregar el punto "trampa" (5,5) como Clase A

# 1. Crear el dataset (X = Coordenadas, Y = Etiquetas binarias 0 o 1)
X = np.array([ [2,2], [3,3], [4,2], [6,6], [7,8], [8,7], [5,5] ])
Y = np.array([ 0, 0, 0, 1, 1, 1, 0 ])  # el nuevo punto (5,5) es Clase A

modelo_svm_lineal = SVC(kernel='linear')
modelo_svm_lineal.fit(X, Y)

# 3. Entrenar el modelo (Aprender la ecuación del hiperplano)
modelo_svm.fit(X, Y)

# 4. Extraer los Vectores de Soporte descubiertos por la IA
vectores = modelo_svm.support_vectors_
print("Los Vectores de Soporte son:\n", vectores)

# 5. Predicción
nuevo_punto = np.array([[5, 4]])
pred = modelo_svm.predict(nuevo_punto)
print("El punto [5,4] pertenece a la clase:", pred[0])


# Punto 4: Cambiar a kernel RBF

X = np.array([ [2,2], [3,3], [4,2], [6,6], [7,8], [8,7], [5,5] ])
Y = np.array([ 0, 0, 0, 1, 1, 1, 0 ])  # el nuevo punto (5,5) es Clase A

modelo_svm_rbf = SVC(kernel='rbf')
modelo_svm_rbf.fit(X, Y)
# 3. Entrenar el modelo (Aprender la ecuación del hiperplano)
#modelo_svm.fit(X, Y)

# 4. Extraer los Vectores de Soporte descubiertos por la IA
vectores = modelo_svm.support_vectors_
print("Los Vectores de Soporte son:\n", vectores)

# 5. Predicción
nuevo_punto = np.array([[5, 4]])
pred = modelo_svm_rbf.predict(nuevo_punto)
print("Prediccion con kernel RBF:", pred[0])




# Punto 5 — Reflexión: ¿Cuándo falla completamente un kernel lineal?

# el reconocimiento facial: las variaciones de iluminación, ángulo y expresión
# hacen que los rostros de una misma persona no formen un grupo compacto y
# linealmente separable en el espacio de características — se necesitan fronteras
# curvas y complejas (RBF u otros kernels no lineales) para capturar esa variabilidad.

