from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import cv2

def cargar_imagen(ruta):
    imagen = Image.open(ruta)
    plt.imshow(imagen)
    plt.title("Imagen original")
    plt.axis("off")
    plt.show()
    return imagen

def separar_rgb(imagen):
    r, g, b = imagen.split()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap='Reds')
    plt.title("Componente R")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='Greens')
    plt.title("Componente G")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap='Blues')
    plt.title("Componente B")
    plt.axis("off")
    plt.show()

def convertir_grises(imagen):
    imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
    plt.imshow(imagen_cv, cmap='gray')
    plt.title("Imagen en escala de grises")
    plt.axis("off")
    plt.show()
    return imagen_cv

def binarizar_imagen(imagen_gris, umbral=128):
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    plt.imshow(binaria, cmap='gray')
    plt.title(f"Imagen binarizada (umbral= {umbral})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    ruta_imagen = "imagen_ejemplo.jpg"
    imagen = cargar_imagen(ruta_imagen)
    separar_rgb(imagen)
    imagen_gris = convertir_grises(imagen)
    binarizar_imagen(imagen_gris, umbral=128)