import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Directorios de datasets
ruta_base_datos = '/datasets_residuos'
categorias = ['papel', 'plastico', 'organico']

# Carga de datos de entrenamiento
datos = []
etiquetas = []

for categoria in categorias:

    ruta_imgs = os.path.join(ruta_base_datos, categoria)

    for img in os.listdir(ruta_imgs):

        # Leo imagen
        ruta_img = os.path.join(ruta_imgs, img)
        img_array = cv2.imread(ruta_img)

        # Agrego a datos
        datos.append(img_array)

        # Defino etiqueta segun carpeta
        if categoria == 'papel':
            etiquetas.append(0)
        elif categoria == 'plastico':
            etiquetas.append(1)
        else:
            etiquetas.append(2)

datos = np.array(datos)
etiquetas = np.array(etiquetas)

# Defino y entreno el modelo
modelo = Sequential()
modelo.add(Conv2D(32, 3, 3, activation='relu', input_shape=(150, 150, 3)))
modelo.add(Flatten())
modelo.add(Dense(3, activation='softmax'))

modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

modelo.fit(datos, etiquetas, epochs=10)

modelo.save('modelo_custom.h5')