
#"------------Taller Analítico: Contando Parámetros-------------"

# Pregunta 1: Pesos (W) entre la Capa de Entrada y la Capa Oculta

# Cada una de las 3 entradas se conecta con cada una de las 4 neuronas ocultas:
# 3 entradas × 4 neuronas ocultas = 12 pesos



# Pregunta 2: Sesgos en la Capa Oculta

# Cada neurona tiene su propio sesgo, independientemente de cuántas entradas reciba:
# 4 neuronas ocultas → 4 sesgos


# Pregunta 3: Pesos y Sesgo entre la Capa Oculta y la Capa de Salida

# Pesos: 4 neuronas ocultas × 1 neurona de salida = 4 pesos
# Sesgos: 1 neurona de salida → 1 sesgo


# Pregunta 4: Total de parámetros entrenables

# Pesos capa 1:   12
# Sesgos capa 1:   4
# Pesos capa 2:    4
# Sesgos capa 2:   1
# ─────────────────────
# TOTAL:          21 parámetros entrenables




# "-------------------Taller de Laboratorio: Explorando las Matrices------------"


import numpy as np

# Funcion de Activacion: Sigmoide (devuelve un valor entre 0 y 1)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# 1. ENTRADA (X): 1 cliente con 3 caracteristicas
X = np.array([0.5, 0.8, 0.2])

# 2. CAPA OCULTA (4 Neuronas)
# Matriz W1 de (3 entradas x 4 neuronas)
W1 = np.array([
    [0.1,  0.2, -0.3,  0.4],
    [-0.5, 0.6,  0.7, -0.8],
    [0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, -0.2, 0.3, -0.4])  # 4 Sesgos

# --- PROCESO CAPA OCULTA ---
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)  # Salida de la capa oculta

# 3. CAPA DE SALIDA (1 Neurona)
# Matriz W2 de (4 entradas ocultas x 1 neurona final)
W2 = np.array([0.5, -0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# --- PROCESO CAPA FINAL ---
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("Prediccion de la Red (Probabilidad):", np.round(Salida_Final[0], 4))
print("Z1 (valores puros):", Z1)
print("A1 (despues de sigmoide):", A1)

# Punto 1 y 2: Al correr el código, Se obtiene una probabilidad específica,
# y al imprimir Z1 y A1 verás cómo la sigmoide "comprime" los valores:
# Z1 puede tener valores negativos o mayores a 1 (por ejemplo, algo como
# [0.11, 0.42, -0.18, 0.31]), mientras que A1 siempre queda comprimido dentro del
# rango (0, 1) — por ejemplo [0.527, 0.603, 0.455, 0.577]



# Punto 3 y 4: El Reto Dimensional — Procesar 2 clientes a la vez (Batch)


def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# 1. ENTRADA (X): ahora 2 clientes con 3 caracteristicas cada uno
X = np.array([
    [0.5, 0.8, 0.2],   # Cliente 1
    [0.1, 0.9, 0.9]    # Cliente 2
])

# 2. CAPA OCULTA (los pesos NO cambian)
W1 = np.array([
    [0.1,  0.2, -0.3,  0.4],
    [-0.5, 0.6,  0.7, -0.8],
    [0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, -0.2, 0.3, -0.4])

Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)

# 3. CAPA DE SALIDA (tampoco cambia)
W2 = np.array([0.5, -0.6, 0.7, 0.8])
b2 = np.array([-0.1])

Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("Predicciones de la Red (2 clientes):", np.round(Salida_Final, 4))

# no tuviste que tocar W1, b1, W2 ni b2 en absoluto. Las mismas matrices de pesos,
# sin ningún cambio, procesaron 1 cliente o 2 clientes con solo cambiar la forma de X.
# Esto es exactamente por qué las redes neuronales reales procesan miles de imágenes
# a la vez (un "batch" o lote) en una sola operación de GPU, en vez de una por una
# — el álgebra matricial escala automáticamente sin cambiar el modelo.