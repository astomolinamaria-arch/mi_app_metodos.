%%writefile app.py
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Clasificación Termográfica", layout="wide")

st.title("Clasificación Termográfica de Fallas en Pasteurización")
st.write("Aplicación interactiva para el análisis de gradiente térmico de enfriamiento post-pasteurización.")

# Menú en la barra lateral
st.sidebar.header("Cargar Imagen")
archivo_subido = st.sidebar.file_uploader("Sube una imagen térmica...", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    # Leer la imagen subida por el usuario
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    imagen_np = np.array(imagen_pil)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Imagen Original")
        st.image(imagen_pil, use_container_width=True)

    with col2:
        st.subheader("Análisis de Métodos Numéricos (Diferencia Centrada)")
        
        # Convertir a escala de grises
        img_gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h = 1
        dI_dx_centered = np.zeros_like(img_gris)
        
        # Aplicar diferencias finitas centradas
        for y in range(1, img_gris.shape[0] - 1):
            for x in range(1, img_gris.shape[1] - 1):
                dI_dx_centered[y, x] = (img_gris[y, x + h] - img_gris[y, x - h]) / (2 * h)

        # Graficar gradiente
        fig, ax = plt.subplots()
        im = ax.imshow(np.abs(dI_dx_centered), cmap='jet')
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        ax.set_title("Gradiente Térmico Aprox.")
        st.pyplot(fig)
else:
    st.info(" Sube una imagen térmica desde la barra lateral izquierda para iniciar el análisis.")
