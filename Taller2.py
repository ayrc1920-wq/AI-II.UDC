import cv2
import numpy as np

imagen = cv2.imread('muestra.jpg')

# Estraccion de de matrice 2D individuales (cilicing)
canal_azul = imagen[:, :, 0]
canal_verde = imagen[:, :, 1]
canal_rojo = imagen[:, :, 3]

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