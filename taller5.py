# Pregunta 1: Convolución con Sobel X (Gx),centrado en el píxel del medio

# (0×-1) + (0×0) + (255×1)  =   0 +   0 + 255 =  255
# (0×-2) + (0×0) + (255×2)  =   0 +   0 + 510 =  510
# (0×-1) + (0×0) + (255×1)  =   0 +   0 + 255 =  255

# Suma total = 255 + 510 + 255 = 1020

# Un valor muy alto y positivo — esto confirma matemáticamente que
# hay un cambio brusco de intensidad en la dirección horizontal
# : exactamente lo que esperamos, ya que la mitad izquierda es negra
# (0) y la derecha es blanca (255).

# Pregunta 2: Convolución con Sobel Y (Gy). ¿Por qué el resultado es cero? 

# (0×-1) + (0×-2) + (255×-1) =   0 +   0 - 255 = -255
# (0×0)  + (0×0)  + (255×0)  =   0 +   0 +   0 =    0
# (0×1)  + (0×2)  + (255×1)  =   0 +   0 + 255 =  255

# Suma total = -255 + 0 + 255 = 0

#Por qué da cero: el kernel Gy compara los valores de arriba contra abajo
# (fila superior vs fila inferior). Pero en esta matriz, las tres filas son idénticas
# (0, 0, 255 se repite exactamente igual en cada fila) — no hay ningún cambio de
# intensidad al moverse verticalmente.

# Qué indica sobre la dirección del borde: un Gy = 0 significa que el borde no tiene
# ninguna componente horizontal — es decir, el borde es puramente vertical. Esto tiene
# sentido geométricamente: un borde vertical (una línea que separa izquierda de derecha)
# only produces change of intensity when you move horizontally (for that Gx is large),
# but if you move vertically along that same line, the intensity does not change at all
# (for that Gy is exactly 0).




#"-------------Taller de Laboratorio: Inspector de Bordes---------------"

import cv2
import numpy as np

img1 = cv2.imread('images (1).jpeg', cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_vis = cv2.convertScaleAbs(sobelx)

sobely = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_vis = cv2.convertScaleAbs(sobely)

bordes_canny = cv2.Canny(img1, 200, 250)

panel = np.hstack([sobel_x_vis, sobel_y_vis, bordes_canny])
cv2.imshow('Sobel X | Sobel Y | Canny', panel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Con umbrales bajos (10, 50): el algoritmo es muy permisivo — casi cualquier variación
# de intensidad se acepta como borde, incluyendo ruido y texturas menores (pasto, ladrillos,
# granulado de la foto). El resultado se ve "sucio", con muchísimos bordes falsos.

# Con umbrales altos (200, 250): el algoritmo es muy estricto — solo los cambios de intensidad
# más extremos se consideran bordes. Se pierden contornos reales pero sutiles
# (sombras suaves, texturas), y solo sobreviven los contornos más marcados
# (la silueta principal de un edificio contra el cielo, por ejemplo).

# El umbral "óptimo" para una imagen específica está en un punto intermedio, y
# se encuentra empíricamente probando distintos valores hasta que los bordes reales del
# objeto de interés queden marcados sin demasiado ruido de fondo.




