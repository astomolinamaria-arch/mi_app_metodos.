import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# Configuración de la página
st.set_page_config(page_title="Clasificación Termográfica", layout="wide")

st.title("Clasificación Termográfica de Fallas en Pasteurización")
st.write("Aplicación interactiva para el análisis de gradiente térmico de enfriamiento post-pasteurización.")

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('modelo_pasteurizacion.h5')

try:
    modelo = cargar_modelo()
    # Clases ordenadas alfabéticamente tal como las crea TensorFlow desde carpetas
    clases = ['Enfriamiento Deficiente', 'Enfriamiento Óptimo', 'Sobreenfriamiento']
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Menú en la barra lateral
st.sidebar.header("Cargar Imagen")
archivo_subido = st.sidebar.file_uploader("Sube una imagen térmica...", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    imagen_np = np.array(imagen_pil)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen Original")
        st.image(imagen_pil, use_container_width=True)

    with col2:
        st.subheader("Análisis de Métodos Numéricos (Diferencia Centrada)")
        img_gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h = 1
        dI_dx_centered = np.zeros_like(img_gris)

        for y in range(1, img_gris.shape[0] - 1):
            for x in range(1, img_gris.shape[1] - 1):
                dI_dx_centered[y, x] = (img_gris[y, x + h] - img_gris[y, x - h]) / (2 * h)

        fig, ax = plt.subplots()
        im = ax.imshow(np.abs(dI_dx_centered), cmap='jet')
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        ax.set_title("Gradiente Térmico Aprox.")
        st.pyplot(fig)

    # --- CLASIFICACIÓN CON MODELO DE IA ---
    st.markdown("---")
    st.subheader("Resultado de la Clasificación (Red Neuronal CNN)")

    # Preprocesamiento exacto como en Colab (64x64 píxeles)
    img_resized = cv2.resize(imagen_np, (64, 64))
    img_array = np.expand_dims(img_resized, axis=0) # Crear batch de 1 imagen

    # Predicción
    prediccion = modelo.predict(img_array)
    indice = np.argmax(prediccion)
    categoria_predicha = clases[indice]
    confianza = np.max(prediccion) * 100

    # Mostrar respuesta dinámica según el resultado
    if categoria_predicha == 'Enfriamiento Deficiente':
        st.error(f"⚠️ **Categoría:** {categoria_predicha} (Confianza: {confianza:.1f}%)")
    elif categoria_predicha == 'Enfriamiento Óptimo':
        st.success(f"✅ **Categoría:** {categoria_predicha} (Confianza: {confianza:.1f}%)")
    else:
        st.warning(f"❄️ **Categoría:** {categoria_predicha} (Confianza: {confianza:.1f}%)")

else:
    st.info("Sube una imagen térmica desde la barra lateral izquierda para iniciar el análisis.")
