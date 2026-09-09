#"------------Taller analitico-----------
# 1 Preg
# RPT: A2,3 = 0. Como el índice inicia en 0, corresponde a la tercera fila y cuarta columna
#  de la matriz. El valor 0 representa un píxel negro en la representación visuaL 

# 2 Preg
# RPT:Un tensor RGB de 1080 X 1920 X 3 contiene 1080 X 1920 píxeles y 3 valores por cada píxel: rojo, verde y azul. Como cada valor ocupa 1 byte,
# se almacenan 1080 X 1920 X 3 = 6.220.800 bytes, aproximadamente 6,22 MB sin compresión. 

import numpy as np
# matriz original 3x3 muy oscura
A = np.array([
    [10, 20, 10,],
    [15, 30, 15,],
    [10, 20, 10,]
], dtype=np.float32)

# Parametros lineales
alpha = 2.5       # Aumento  de contraste (multiplicacion escalar)
beta = 50.0       # Aumento de brillo (suma escalar a todos los elementos)

# Transformacion linear : Anueva = a*A + b
A_nueva = (alpha * A) + beta

# Acotamiento (clipping) para mantener formato de imagen de 8 bits
A_nueva = np.clip(A_nueva, 0, 255).astype(np.uint8)

print("Matriz resultante :")
print(A_nueva)


#"-----------Taller de laboratorio---------"

# Matriz de prueba 5x5, valores sobreexpuestos entre 200 y 254
np.random.seed
A = np.random.randint(200,255, (5,5))

# Parámetros de la transformación
Alpha = 0.5     # reduce el contraste al 50% -> "aplasta" el rango de valores
Beta = -50    # disminuye el brillo en 50 unidades -> se RESTA, no se suma

A_procesada = (Alpha * A) + Beta

A_procesada = np.clip(A_procesada, 0, 255).astype(np.uint8)

print("Matriz original:\n", A)
print("\nMatriz procesada:\n", A_procesada)





#"------------Taller analitico-----------"


A = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

# Transpuesta usando NumPy
A_transpuesta = A.T

# Aplanamiento (Vectorización para Redes Neuronales)
vector_1D = A.flatten()

print("Forma original:", A.shape)                # Salida: (2, 3)
print("Forma transpuesta:", A_transpuesta.shape) # Salida: (3, 2)
print("Vector plano:", vector_1D)                # Salida: [10 20 30 40 50 60]


# 1 Preg
# RPT:El resultado es exactamente la misma matriz I. Geométricamente: la identidad solo
# tiene valores distintos de cero en la diagonal principal,
# y transponer intercambia elementos "espejo" respecto a esa diagonal

# 2 Preg
# RPT:Cada píxel de cada canal se convierte en una entrada individual del vector,
# así que la capa de entrada debe tener exactamente ese número de neuronas para poder
# recibir el vector completo sin perder información.


#"-----------Taller de laboratorio final---------"


# Definimos la sección de imagen y el kernel de realce (sharpen)
I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
])

K = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

#  Producto Hadamard: multiplicación elemento a elemento
producto = I * K
print("Producto Hadamard (elemento a elemento):\n", producto)

#  Suma total de la matriz resultante = valor del nuevo píxel central
pixel_central = np.sum(producto)
print("Valor del píxel central calculado:", pixel_central)