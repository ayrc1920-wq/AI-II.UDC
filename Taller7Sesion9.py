#"-------------TALLER ANALITICO----------"

# A(20, 30) → NO COMPRA
# B(40, 50) → COMPRA
# C(35, 45) → COMPRA

# NUEVO CLIENTE
# D (30, 40) -> ?

# Pregunta 1: Calcular la distancia del punto nuevo hasta A, B y C

# 𝑑(Nuevo,A)= √(30− 20)2 + (40− 30)2 = √100+ 100 = √200 ≈ 14.14

# 𝑑(Nuevo,B)= √(30− 40)2+ (40− 50)2 = √100+ 100 = √200 ≈ 14.14

# 𝑑(Nuevo,C)= √(30− 35)2+ (40− 45)2 = √25+ 25 = √50 ≈ 7.07

# Pregunta 2: Con K = 1, ¿cuál es la clasificación?
# Con K=1 solo se consulta al vecino más cercano. La distancia más pequeña es
# hacia C (7.07), y C es COMPRA.
# Clasificación con K=1: COMPRA


# Pregunta 3: Con K = 3, ¿cuál es la clasificación? ¿Hubo cambio?
# Con K=3 se consultan los tres puntos (es todo el dataset disponible), y se vota por mayoría:

# A → NO COMPRA (1 voto)
# B → COMPRA (1 voto)
# C → COMPRA (1 voto)

# Total: 2 votos COMPRA vs 1 voto NO COMPRA.
# Clasificación con K=3: COMPRA

# No hubo cambio en la decisión — en ambos casos (K=1 y K=3) el resultado es COMPRA.




#"---------------TALLER DE LABORATORIO--------------"
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X_entrenamiento = np.array([
[20, 30], # Punto A
[40, 50], # Punto B
[35, 45] # Punto C
])
# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1])
# 2. Instanciar el modelo con K = 3
modelo_knn = KNeighborsClassifier(n_neighbors=3)
# 3. "Entrenar" (Memorizar los datos)
modelo_knn.fit(X_entrenamiento, Y_entrenamiento)
# 4. Predecir un nuevo punto
nuevo_cliente = np.array([[30, 40]])
prediccion = modelo_knn.predict(nuevo_cliente)
print("Clase predicha:", prediccion[0])



#"-------------Taller de Laboratorio: Clasificador Universal-------"

X_entrenamiento = np.array([
[20, 30, 0], # No compra
[22, 32, 1], # No compra
[25, 35, 0], # No compra
[23, 28, 0], # No compra
[28, 33, 1], # No compra
[40, 50, 2], # Compra
[42, 55, 1], # Compra
[35, 45, 1], # Compra
[45, 60, 3], # Compra
[38, 48, 2], # Compra
])

Y_entrenamiento = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

nuevo_cliente = np.array([[30, 40, 1]])

modelo_k1 = KNeighborsClassifier(n_neighbors=1)
modelo_k1.fit(X_entrenamiento, Y_entrenamiento)
print("Prediccion con K=1:", modelo_k1.predict(nuevo_cliente)[0])

modelo_k5 = KNeighborsClassifier(n_neighbors=5)
modelo_k5.fit(X_entrenamiento, Y_entrenamiento)
print("Prediccion con K=5:", modelo_k5.predict(nuevo_cliente)[0])



#"---------------Punto 5 — Pregunta de Análisis:--------"

# Las distancias pierden significado (fenómeno de concentración): cuando sumas
# cientos o miles de diferencias al cuadrado, casi todos los puntos terminan teniendo distancias
# muy parecidas entre sí — la diferencia relativa entre “el vecino más cercano” y “el
# vecino más lejano” se vuelve estadísticamente insignificante. KNN depende completamente
# de que existan vecinos claramente “más cercanos” que otros; si todos los puntos parecen casi
# equidistantes, el algoritmo pierde su capacidad de discriminar.

# El espacio se vuelve extremadamente disperso (sparse): para llenar razonablemente un
# espacio de 1,000 dimensiones necesitarías una cantidad astronómica de datos — muchísimos
# más que los que normalmente tienes disponibles. Con pocos datos en un espacio tan grande,
# cualquier punto nuevo termina estando “lejos de todo”, porque los puntos memorizados están
# dispersos en un volumen inmenso comparado con la cantidad de ejemplos que tienes.

# Costo computacional: cada cálculo de distancia ahora involucra 1,000 restas y 1,000
# elevaciones al cuadrado en vez de 3 — y recuerda que KNN hace esto contra todos los
# puntos memorizados en cada predicción (es “perezoso”), así que el costo crece linealmente
# con el número de dimensiones y con el tamaño del dataset simultáneamente.



