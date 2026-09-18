#"-----------Taller Analítico: Calculando el Disparo--------"

# Pregunta 1: Calcular Z

#Pregunta 1: Calcular Z

# Z = (X1 · W1) + (X2 · W2) + b
# Z = (50 × 0.8) + (20 × -0.5) + (-10)
# Z = 40 + (-10) + (-10)
# Z = 20

# Pregunta 2: ¿La neurona aprueba o rechaza el crédito?

# Aplicando la Función Escalón: como Z = 20 es mayor o igual a 0, la neurona dispara 1.
# → La neurona APRUEBA el crédito (Salida = 1).

# Pregunta 3: ¿Por qué tiene sentido que W2 sea negativo?

# En este escenario empresarial, W1 (peso de los Ingresos) es positivo porque a mayor
#  ingreso, mayor debería ser la probabilidad de aprobación — el ingreso "empuja" a Z
#  hacia arriba, favoreciendo el disparo.
# W2 (peso de las Deudas) es negativo porque la relación es inversa: entre más deudas
# tenga el cliente, menor debería ser la probabilidad de que se apruebe el crédito.
# Al ser negativo, cada unidad de deuda.
# resta valor a Z, empujando a la neurona hacia el rechazo (0). Esto refleja matemáticamente
# el criterio de riesgo: ingresos altos ayudan a aprobar, deudas altas ayudan a rechazar
# — cada peso codifica si esa característica favorece o perjudica la decisión final.



#"--------------Taller de Laboratorio: Hackeando los Pesos — La Compuerta OR-----------"

import numpy as np
 
# 1. Definir la Funcion de Activacion (Escalon)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0
 
# 2. Definir la Estructura de la Neurona
def perceptron(X, W, b):
    # Producto punto (Combinacion lineal)
    # Equivalente a: (X[0]*W[0]) + (X[1]*W[1]) ...
    Z = np.dot(X, W) + b
 
    # Activacion
    salida = funcion_escalon(Z)
    return salida
 
# 3. Datos del problema (Compuerta Logica AND)
# El AND solo da 1 si ambas entradas son 1.
entradas = np.array([1, 1])   # Vector X
pesos = np.array([0.5, 0.5])  # Vector W
sesgo = -0.8                  # Constante b
 
# 4. Inferencia (Forward pass)
resultado = perceptron(entradas, pesos, sesgo)
print("El Perceptron disparo el valor:", resultado)

#"-------Taller de Laboratorio: Hackeando los Pesos---------"

# Pregunta 1:
# entradas = np.array([1, 1]) con pesos = [0.5, 0.5] y sesgo = -0.8

# Z = (1×0.5)+(1×0.5)+(-0.8) = 1 - 0.8 = 0.2. Como 0.2 ≥ 0, la salida es 1. Esto confirma
# la compuerta AND: solo cuando ambas entradas son 1 el resultado supera el sesgo negativo
# y la neurona dispara.




# Punto 2: Verificar que el código actual (AND) rechaza los demás casos

# pesos = np.array([0.5, 0.5])
# sesgo = -0.8
 
# print(perceptron(np.array([1, 0]), pesos, sesgo))  # Z = 0.5 - 0.8 = -0.3 -> 0
# print(perceptron(np.array([0, 1]), pesos, sesgo))  # Z = 0.5 - 0.8 = -0.3 -> 0
# print(perceptron(np.array([0, 0]), pesos, sesgo))  # Z = 0.0 - 0.8 = -0.8 -> 0

# con los pesos del AND, las tres combinaciones ([1,0], [0,1], [0,0]) dan Z negativo
# y la neurona arroja 0 en los tres casos

# Punto 3 y 4: Encontrar los pesos y el sesgo para la Compuerta OR

# La Compuerta OR debe dar 1 si al menos una entrada es 1, y solo 0 cuando ambas
#  son 0. Una combinación que resuelve esto matemáticamente es:
# W1 = 1     W2 = 1     b = -0.5

# Código final para verificarlo en Python:
# pesos_or = np.array([1, 1])
# sesgo_or = -0.5
 
# print(perceptron(np.array([0, 0]), pesos_or, sesgo_or))  # 0
# print(perceptron(np.array([1, 0]), pesos_or, sesgo_or))  # 1
# print(perceptron(np.array([0, 1]), pesos_or, sesgo_or))  # 1
# print(perceptron(np.array([1, 1]), pesos_or, sesgo_or))  # 1



# Punto 5: Reflexión
# Los pesos W1 = 1, W2 = 1 y el sesgo b = -0.5 son exactamente lo que "aprendería"
# una red neuronal si se entrenara automáticamente para resolver este mismo problema
#  por ensayo y error / razonamiento matemático, mientras que en la próxima sesión
#  un algoritmo de
#  entrenamiento (descenso de gradiente) ajustará estos valores de forma automática
# , probando y corrigiendo iterativamente hasta encontrar una combinación que funcione
# . Este ejercicio manual es justamente el puente conceptual: entender qué es lo que el
# algoritmo de entrenamiento va a estar buscando

