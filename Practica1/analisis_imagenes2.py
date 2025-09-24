# -*- coding: utf-8 -*-
from collections import OrderedDict
from tkinter import Tk, filedialog
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")  # Asegúrate de tener PyQt5 instalado
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2
from skimage import color, util


# ========= utilidades generales =========
def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convierte cualquier imagen a uint8 sin alterar su apariencia visual."""
    if img.dtype == np.uint8:
        return img
    if img.dtype.kind == "f" and 0.0 <= float(img.min()) and float(img.max()) <= 1.0:
        return (img * 255).clip(0, 255).astype(np.uint8)
    m, M = float(img.min()), float(img.max())
    if M - m < 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - m) / (M - m) * 255.0).astype(np.uint8)


def plot_hist_on_axes(ax, img_u8: np.ndarray, name: str):
    """Dibuja un histograma: 1 curva si es gris, 3 curvas si es RGB."""
    ax.set_title(name)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=.3)

    if img_u8.ndim == 2:  # gris
        hist, bins = np.histogram(img_u8.ravel(), bins=256, range=(0, 255))
        ax.plot(bins[:-1], hist)
    else:  # RGB
        for i, c in enumerate(("red", "green", "blue")):
            hist, bins = np.histogram(img_u8[..., i].ravel(), bins=256, range=(0, 255))
            ax.plot(bins[:-1], hist, label=c, color=c)
        ax.legend()


def open_hist_grid(images_dict: OrderedDict, title: str = "Histogramas", hist_whitelist=None):
    """
    Abre UNA sola ventana con un grid de histogramas SOLO para los nombres
    incluidos en 'hist_whitelist'. Si es None, usa todos.
    """
    # Filtra por whitelist si se especifica
    if hist_whitelist is not None:
        items = [(k, v) for k, v in images_dict.items() if k in hist_whitelist]
    else:
        items = list(images_dict.items())

    if not items:
        print("No hay imágenes seleccionadas para histogramas.")
        return

    names, imgs = zip(*items)
    n = len(imgs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    for ax, name, img in zip(axes, names, imgs):
        u8 = to_uint8(img)
        plot_hist_on_axes(ax, u8, name)

    # Apaga ejes sobrantes si el grid no es exacto
    for ax in axes[len(imgs):]:
        ax.axis("off")

    fig.suptitle(title, y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=True)


def figure_with_hist_button(images_dict: OrderedDict, title=None, layout=(1, 1), cmaps=None, hist_whitelist=None):
    """
    Muestra images_dict en una figura y agrega un botón "Histogramas".
    El botón abrirá SOLO los histogramas de las claves en 'hist_whitelist'.
    Si 'hist_whitelist' es None, graficará todos los items.
    """
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=(11, 6))
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    # Deja espacio inferior para el botón
    fig.subplots_adjust(bottom=0.18, top=0.90)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    items = list(images_dict.items())
    for ax, (name, img) in zip(axes, items):
        if isinstance(img, np.ndarray) and img.ndim == 2:
            ax.imshow(img, cmap=(cmaps or {}).get(name, "gray"))
        else:
            ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    # Botón (solo histogramas de ESTA figura, en una sola ventana)
    ax_btn = fig.add_axes([0.73, 0.04, 0.24, 0.08])
    btn = Button(ax_btn, "Histogramas")

    def _on_click(_event):
        open_hist_grid(
            images_dict,
            title=f"Histogramas — {title or 'Figura'}",
            hist_whitelist=hist_whitelist
        )

    # Guardar referencias para evitar garbage collection
    fig._hist_btn = btn
    fig._hist_cid = btn.on_clicked(_on_click)
    fig._hist_images = images_dict
    fig._hist_whitelist = hist_whitelist
    return fig, axes


# ========= pasos base =========
def cargar_rgb(path: str) -> np.ndarray:
    """Carga con PIL en RGB uint8."""
    pil = Image.open(path).convert("RGB")
    return np.array(pil)


def separar_canales(rgb: np.ndarray):
    """Devuelve R, G, B (2D cada uno)."""
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def a_grises_bt601(rgb: np.ndarray) -> np.ndarray:
    """Escala de grises por luminancia (BT.601) con OpenCV."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def binarizar(gray_u8: np.ndarray, t: int = 128) -> np.ndarray:
    """Umbral fijo."""
    _, binaria = cv2.threshold(gray_u8, t, 255, cv2.THRESH_BINARY)
    return binaria


# ========= nuevos modelos de color =========
def to_float01(rgb_u8: np.ndarray) -> np.ndarray:
    """Convierte RGB uint8 a float32 [0,1] para skimage.color."""
    return util.img_as_float32(rgb_u8)


def convertir_yiq(rgb_u8: np.ndarray) -> OrderedDict:
    """
    YIQ (skimage): devuelve Y, I, Q como 2D y una vista 'visual' (Y/I/Q apilados para mostrar).
    Para histogramas usaremos SOLO Y, I, Q (no la 'visual').
    """
    rgb_f = to_float01(rgb_u8)
    yiq = color.rgb2yiq(rgb_f)  # float [0,1] para Y; I/Q en rangos aprox. [-0.6, 0.6]
    Y, I, Q = yiq[..., 0], yiq[..., 1], yiq[..., 2]

    # Reescala I y Q a [0,1] para visualización
    def rescale01(x):
        xmin, xmax = float(x.min()), float(x.max())
        if xmax - xmin < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return (x - xmin) / (xmax - xmin)

    Y_u8 = to_uint8(Y)
    I_u8 = to_uint8(rescale01(I))
    Q_u8 = to_uint8(rescale01(Q))
    yiq_vis = np.dstack([Y_u8, I_u8, Q_u8])  # solo para ver en color, NO para histograma

    return OrderedDict({
        "YIQ - Y (luminancia)": Y_u8,
        "YIQ - I (crominancia)": I_u8,
        "YIQ - Q (crominancia)": Q_u8,
        "YIQ (visual)": yiq_vis  # <- mostrado en figura, ignorado en histograma
    })


def convertir_cmy(rgb_u8: np.ndarray) -> OrderedDict:
    """
    CMY sustractivo simple: C=255-R, M=255-G, Y=255-B.
    Devuelve canales y un composite para visualizar.
    """
    cmy_u8 = 255 - rgb_u8
    C, M, Y = cmy_u8[..., 0], cmy_u8[..., 1], cmy_u8[..., 2]
    return OrderedDict({
        "CMY - C (255-R)": C,
        "CMY - M (255-G)": M,
        "CMY - Y (255-B)": Y,
        "CMY (composite)": cmy_u8  # <- mostrado en figura, ignorado en histograma
    })


def convertir_hsv(rgb_u8: np.ndarray) -> OrderedDict:
    """
    HSV con skimage: H,S,V en [0,1] + una reconstrucción a RGB para ver colores.
    """
    rgb_f = to_float01(rgb_u8)
    hsv = color.rgb2hsv(rgb_f)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H_u8, S_u8, V_u8 = to_uint8(H), to_uint8(S), to_uint8(V)
    hsv_to_rgb_u8 = to_uint8(color.hsv2rgb(hsv))
    return OrderedDict({
        "HSV - H (matiz)": H_u8,
        "HSV - S (saturación)": S_u8,
        "HSV - V (valor)": V_u8,
        "HSV→RGB (reconstruido)": hsv_to_rgb_u8  # <- mostrado en figura, ignorado en histograma
    })


# ========= demo secuencial (una ventana a la vez) =========
def demo(path="imagen.jpg", umbral=128):
    # 1) Original
    rgb = cargar_rgb(path)
    figure_with_hist_button(
        OrderedDict({"Original RGB": rgb}),
        title="1) Original",
        layout=(1, 1),
        hist_whitelist={"Original RGB"}  # histograma RGB del original
    )
    plt.show(block=True)

    # 2) Separada en canales RGB
    r, g, b = separar_canales(rgb)
    figure_with_hist_button(
        OrderedDict({"Canal R": r, "Canal G": g, "Canal B": b}),
        title="2) Canales RGB",
        layout=(1, 3),
        cmaps={"Canal R": "Reds", "Canal G": "Greens", "Canal B": "Blues"},
        hist_whitelist={"Canal R", "Canal G", "Canal B"}
    )
    plt.show(block=True)

    # 3) Escala de grises (BT.601)
    gray = a_grises_bt601(rgb)
    figure_with_hist_button(
        OrderedDict({"Escala de grises (BT.601)": gray}),
        title="3) Escala de grises",
        layout=(1, 1),
        hist_whitelist={"Escala de grises (BT.601)"}
    )
    plt.show(block=True)

    # 4) Binarización
    binaria = binarizar(gray, umbral)
    figure_with_hist_button(
        OrderedDict({f"Binarización (t={umbral})": binaria}),
        title="4) Binarización",
        layout=(1, 1),
        hist_whitelist={f"Binarización (t={umbral})"}
    )
    plt.show(block=True)

    # 5) YIQ
    yiq_imgs = convertir_yiq(rgb)
    figure_with_hist_button(
        yiq_imgs,
        title="5) Modelo YIQ (aplicado a RGB original)",
        layout=(2, 2),
        hist_whitelist={
            "YIQ - Y (luminancia)",
            "YIQ - I (crominancia)",
            "YIQ - Q (crominancia)",
            # Omitimos "YIQ (visual)"
        }
    )
    plt.show(block=True)

    # 6) CMY
    cmy_imgs = convertir_cmy(rgb)
    figure_with_hist_button(
        cmy_imgs,
        title="6) Modelo CMY (aplicado a RGB original)",
        layout=(2, 2),
        hist_whitelist={
            "CMY - C (255-R)",
            "CMY - M (255-G)",
            "CMY - Y (255-B)",
            # Omitimos "CMY (composite)"
        }
    )
    plt.show(block=True)

    # 7) HSV
    hsv_imgs = convertir_hsv(rgb)
    figure_with_hist_button(
        hsv_imgs,
        title="7) Modelo HSV (aplicado a RGB original)",
        layout=(2, 2),
        hist_whitelist={
            "HSV - H (matiz)",
            "HSV - S (saturación)",
            "HSV - V (valor)",
            # Omitimos "HSV→RGB (reconstruido)"
        }
    )
    plt.show(block=True)

#Abrimos una imagen
def seleccionar_imagen():
    app = QApplication(sys.argv)
    # abre un cuadro de diálogo estándar para escoger archivo
    ruta, _ = QFileDialog.getOpenFileName(
        None,
        "Selecciona una imagen",
        "",
        "Archivos de imagen (*.jpg *.jpeg *.png *.bmp *.tiff)"
    )
    return ruta

if __name__ == "__main__":
    # Abrir explorador para elegir imagen
    ruta = seleccionar_imagen()

    if ruta:
        # Pedir umbral, permitir valor por defecto
        entrada = input("Escribe el umbral que deseas (ENTER = 128): ")
        if entrada.strip() == "":
            umbral = 128
        else:
            try:
                umbral = int(entrada)
            except ValueError:
                print("⚠️ No escribiste un número válido, usaré 128.")
                umbral = 128

        demo(ruta, umbral)

    else:
        print("No seleccionaste ninguna imagen.")

