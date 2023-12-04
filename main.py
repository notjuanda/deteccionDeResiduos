# entrenamiento.py
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split

class PreprocesadoDatos:
    """Clase para preprocesar los datos"""

    def __init__(self):
        self.ruta_base_datos = os.path.join("datos", "entrenamiento")
        self.dimensiones_imagen = (150, 150, 3)
        self.categorias = ['papel', 'vidrio', 'plastico']

    def cargar_datos(self):
        datos = []
        etiquetas = []

        for categoria_id, categoria in enumerate(self.categorias):
            ruta_categoria = os.path.join(self.ruta_base_datos, categoria)

            for imagen in os.listdir(ruta_categoria):
                ruta_imagen = os.path.join(ruta_categoria, imagen)
                img = cv2.imread(ruta_imagen)
                img = cv2.resize(img, self.dimensiones_imagen[:2])
                datos.append(img)
                etiquetas.append(categoria_id)

        return np.array(datos), np.array(etiquetas)

class ModeloCNN:
    """Clase para el modelo convolucional"""

    def __init__(self):
        self.modelo = self.constructor_modelo()

    def constructor_modelo(self):
        modelo = Sequential()
        modelo.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
        modelo.add(MaxPool2D(pool_size=(2, 2)))

        modelo.add(Conv2D(64, (3, 3), activation='relu'))
        modelo.add(MaxPool2D(pool_size=(2, 2)))

        modelo.add(Conv2D(128, (3, 3), activation='relu'))
        modelo.add(MaxPool2D(pool_size=(2, 2)))

        modelo.add(Flatten())
        modelo.add(Dense(128, activation='relu'))
        modelo.add(Dense(3, activation='softmax'))

        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return modelo

    def entrenar(self, datos, etiquetas):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.2, random_state=42, stratify=None)

        # Entrenar el modelo
        self.modelo.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    def guardar(self, ubicacion_pesos):
        self.modelo.save(ubicacion_pesos)

# Programa principal
if __name__ == "__main__":
    proc = PreprocesadoDatos()
    datos, etiquetas = proc.cargar_datos()

    modelo_cnn = ModeloCNN()
    modelo_cnn.entrenar(datos, etiquetas)
    modelo_cnn.guardar(os.path.join("Main", "modelo_entrenado.h5"))
