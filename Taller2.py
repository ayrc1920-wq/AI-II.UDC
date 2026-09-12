import cv2
import numpy as np

imagen = cv2.imread('monedas.jpeg') 

# Estraccion de de matrice 2D individuales (cilicing)
canal_azul = imagen[:, :, 0]
canal_verde = imagen[:, :, 1]
canal_rojo = imagen[:, :, 2]

# Si quisieras "apagar" el canal azul y verde para dejar solo el rojo:

imagen_solo_roja = np.copy(imagen)
imagen_solo_roja[:, :, 0] =0   # Azul a 0
imagen_solo_roja[:, :, 1] = 0   #Verde a 0

#"--------------Taller Analítico 1-------------"

# 1 Preg
# RPT: recorte.shape = (100, 100)
# Contiene específicamente el canal verde de una región cuadrada de 100x100 
# píxeles ubicada en esa zona de la imagen (una porción rectangular, no toda la foto).

# 2 Preg
# RPT: Un tensor de NumPy se almacena como un bloque contiguo de memoria
# (todos los números uno detrás de otro en RAM, no como objetos Python dispersos).
# Cuando usas img[:,:,0], NumPy no recorre píxel por píxel en Python: usa código
# compilado en C que sabe exactamente el "salto" (stride) de memoria que hay que
# dar para llegar a cada valor del canal 0, y en muchos casos ni siquiera copia
# los datos — devuelve una vista (view) que reutiliza la misma memoria con una 
# aritmética de punteros distinta.
# Un ciclo for anidado en Python, en cambio, tiene que: crear un objeto entero de 
# Python en cada iteración, pedirle memoria al intérprete, evaluar 
# bytecode por cada píxel, etc. Para una imagen de 1920x1080 eso 
# son más de 2 millones de iteraciones con overhead de Python en cada 
# una — órdenes de magnitud más lento que la operación vectorizada de NumPy, 
# que delega todo ese trabajo repetitivo al procesador en una sola 
# instrucción masiva (vectorización/SIMD).

#"----------------Taller de Laboratorio 2 -------------"

pixel =np.array([0, 255, 255])

W = np.array([0.114, 0.587, 0.299])

gris = np.dot(pixel, W)

print("Valor en escala de grises:", gris)          # 225.93
print("Redondeado a entero:", round(gris))          # 226



imagen1 = cv2.imread('monedas.webp')
img_gris = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)

cv2.imwrite('monedas.webp', img_gris)



#"----------Taller de laboratorio 2------------"

import matplotlib.pyplot as plt

# 1. Cargar una imagen RGB (OpenCV la carga en orden BGR)
imagen = cv2.imread('images (1).jpeg')

# 2. Separar la imagen en sus 3 canales
canal_azul = imagen[:, :, 0]
canal_verde = imagen[:, :, 1]
canal_rojo = imagen[:, :, 2]

# 3. Calcular el histograma de cada canal por separado
hist_azul  = cv2.calcHist([imagen], [0], None, [256], [0, 256])
hist_verde = cv2.calcHist([imagen], [1], None, [256], [0, 256])
hist_rojo  = cv2.calcHist([imagen], [2], None, [256], [0, 256])

# 4. Graficar los tres histogramas superpuestos
plt.plot(hist_azul, color='blue', label='Azul')
plt.plot(hist_verde, color='green', label='Verde')
plt.plot(hist_rojo, color='red', label='Rojo')

plt.title("Histograma por Canal de Color")
plt.xlabel("Valor del Pixel (0-255)")
plt.ylabel("Frecuencia (Cantidad de píxeles)")
plt.legend()
plt.show()

# La curva que tenga más área hacia la derecha (valores altos, cerca de 255)
# representa el canal con más luz/dominancia en la imagen.
# Por ejemplo, si tomas una foto de un atardecer, probablemente la curva roja esté
# desplazada hacia la derecha (más intensidad) comparada con la azul, indicando que
# el rojo domina la iluminación general.
# Si las tres curvas están muy parecidas y distribuidas de forma similar,
# la imagen tiene una iluminación más neutra/balanceada (blancos y grises).