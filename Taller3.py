#"----------------Taller Analítico 1----------"

#Pregunta 1: Aplicar umbralización binaria con T = 135

# f(x,y) = 255 si I(x,y) >= 135, si no 0:

# 80        no           0,  0,   255
# 120       no           0,  255, 255
# 140       si           0,  0,   255
# 90        no
# 200       si
# 210       si
# 50        no
# 130       no
# 250       si

# Pregunta 2:
# visualmente al objeto: aparece con huecos
# o bordes incompletos como si le faltaran pedazos. Los píxeles con intensidad
# entre 100 y 134 se pierden y se confunden con el fondo



#"--------------- Operaciones Morfológicas:--------------"
import cv2 
import numpy as np 

imagen = cv2.imread('gato.jpeg', cv2.IMREAD_GRAYSCALE)

_, imagen_binaria = cv2.threshold(imagen, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('Original binarizada (con ruido)', imagen_binaria)

kernel = np.ones((3, 3), np.uint8)

erosion = cv2.erode(imagen_binaria, kernel, iterations=1) 
apertura = cv2.dilate(imagen_binaria, kernel, iterations=1)
cv2.imshow('Apertura (Erosion + Dilatacion)', apertura)

dilatacion = cv2.dilate(imagen_binaria, kernel, itertions=1)
cierre = cv2.erode(imagen_binaria, kernel, iterations=1)
cv2.imshow('cierre (dilatacion + erocion)', cierre)

cv2.waitkey(0)
cv2.destroyAllWindows()


# Pregunta 6:
# RPT:
# La opcion 3 resulta mas efectiva en este caso ya que mejora notoriamente la calidad de
# la imagen.