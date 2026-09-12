#"----------------TALLER ANALITICO 1------------"
# Pregunta 1: Calcular el nuevo valor del pixel central (actualmente 250)
# RPT:
# Suma = 10 + 20 + 30 + 15 + 250 + 15 + 20 + 10 + 20 = 390
# Nuevo valor = 390 / 9 = 43.33 = 43

# Pregunta 2: por quE se dice que el filtro de media difumina o suaviza la imagen?
# RPT:
# El valor original de 250 era un pico anOmalo - un "ruido de sal" en medio de una zona
# con valores bajos
# (10 a 30). Al promediarlo con sus 8 vecinos, ese pico se diluye:
# pasO de 250 a apenas 43, quedando mucho mAs cerca del rango de sus vecinos.

import cv2
import numpy as np

#imagen = cv2.imread('imagen_ruidosa.jpg')

# 1. Filtro de Media (Promedio simple 5x5)
# blur_media = cv2.blur(imagen, (5, 5))

# 2. Filtro Gaussiano (Kernel 5x5, desviaciOn estAndar calculada auto)
# blur_gauss = cv2.GaussianBlur(imagen, (5, 5), 0)

# 3. Filtro de Mediana (Excelente para ruido de impulso/Sal y Pimienta)
# Solo recibe un nUmero impar entero para el tamaño del Kernel
# blur_mediana = cv2.medianBlur(imagen, 5)


#"----------------TALLER ANALITICO 2------------"
imagen_ruid = cv2.imread('images.jpeg')
img_media = cv2.blur(imagen_ruid, (7, 7))
img_gauss = cv2.GaussianBlur(imagen_ruid, (7, 7), 0)
img_mediana = cv2.medianBlur(imagen_ruid, 7)

cv2.imshow('original con ruido', imagen_ruid)
cv2.imshow('filtro de media', img_media)
cv2.imshow('filtro de gaussiano', img_gauss)
cv2.imshow('filtro de mediana', img_mediana)

cv2.waitKey(0)
cv2.destroyAllwindows()

# ¿por qué la Mediana ignora los extremos y la
# Media crea manchas grises?
# Filtro de Media: al promediar, un valor extremo (0 o 255) sí participa en la
# suma junto con sus vecinos "normales" (~100). El resultado del promedio queda
# desplazado hacia ese extremo, generando un valor intermedio (ni 100 ni 0/255)
# que se ve como una mancha gris difusa alrededor de donde estaba el punto de ruido —
# el ruido no desaparece, se "esparce" y contamina a sus vecinos.

# Filtro de Mediana: al ordenar todos los valores bajo el kernel, un punto de ruido
# (0 o 255) queda posicionado en uno de los extremos de la lista ordenada
# (es el valor más bajo o el más alto de todo el grupo). Como se elige el valor
# del medio de esa lista, el ruido simplemente nunca es seleccionado como resultado —
# es estadísticamente descartado sin afectar el promedio de sus vecinos. Por eso el
# resultado se ve limpio: el punto de ruido desaparece por completo en vez de convertirse
# en una mancha.

