#"-------------Taller Analítico: Bounding Box---------------"

# Pregunta 1: Determinar X_min, Y_min, X_max, Y_max
# RPT:
# Valores de X: 2, 8, 10, 3   →   X_min = 2    X_max = 10
# Valores de Y: 4, 2, 7, 9    →   Y_min = 2    Y_max = 9

# Pregunta 2: Calcular el ancho (W) y el alto (H)  
# RPT:
# Ancho = X_max - X_min = 10 - 2 = 8
# Alto = Y_max - Y_min = 9 - 2 = 7


#"-------------Taller de Laboratorio: Clasificador de Formas---------------"

import cv2
import numpy as np

img_color = cv2.imread('monedas2.webp' )
img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

_, imagen_binaria = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)
imagen_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

contornos, _ = cv2.findContours(imagen_limpia, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

UMBRAL_TAMANO = 3000 

for i, cnt in enumerate(contornos):
    area = cv2.contourArea(cnt)
    
    if area > 500:
        print(f'Objeto {i}:  Área = {area} pixeles')
        x, y, w, h = cv2.boundingRect(cnt)
        
        if area > UMBRAL_TAMANO:
            color = (0, 0, 255)  # Rojo para objetos grandes
            etiqueta = 'Grande'
        else:
            color = (255, 0, 0)  # Azul para objetos pequeños
            etiqueta = 'Pequeño'
            
        cv2.rectangle(img_color, (x, y), (x + w, y + h), color, 2)  
        
        cv2.putText(img_color, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
cv2.imshow("Clasificador de Formas", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()        