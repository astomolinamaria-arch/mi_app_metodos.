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
    clases = ['Enfriamiento Deficiente', 'Enfriamiento Óptimo', 'Sobreenfriamiento']
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Menú en la barra lateral
st.sidebar.header("Opciones de Análisis")
archivo_subido = st.sidebar.file_uploader("Sube una imagen térmica...", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    imagen_np = np.array(imagen_pil)

    # Validar si la imagen parece ser un mapa térmico válido
    std_color = np.std(imagen_np, axis=(0, 1))
    if np.mean(std_color) < 20:
        st.error("⚠️ La imagen subida no parece ser una termografía válida. Por favor sube una imagen térmica de proceso.")
    else:
        # --- SECCIÓN 1: PROCESAMIENTO VISUAL Y MÉTODOS NUMÉRICOS ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Imagen Original")
            st.image(imagen_pil, use_container_width=True)

        with col2:
            st.subheader("🔥 Gradiente Térmico Aprox. (Diferencia Centrada)")
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
            st.pyplot(fig)

        # --- SECCIÓN 2: MÉTRICAS DEL GRADIENTE Y PERFIL TÉRMICO ---
        st.markdown("---")
        st.subheader("📊 Análisis Métrico y Perfil Térmico Transversal")
        
        promedio_gradiente = np.mean(np.abs(dI_dx_centered))
        max_gradiente = np.max(np.abs(dI_dx_centered))

        col_m1, col_m2 = st.columns(2)
        col_m1.metric(label="Gradiente Promedio ($\partial I / \partial x$)", value=f"{promedio_gradiente:.2f}")
        col_m2.metric(label="Gradiente Máximo Detectado", value=f"{max_gradiente:.2f}")

        # Gráfico de perfil transversal (Línea central)
        centro_y = img_gris.shape[0] // 2
        perfil_linea = img_gris[centro_y, :]

        fig_perfil, ax_p = plt.subplots(figsize=(10, 3))
        ax_p.plot(perfil_linea, color='red', linewidth=2, label='Intensidad Térmica')
        ax_p.axhline(y=np.mean(perfil_linea), color='gray', linestyle='--', label='Promedio')
        ax_p.set_title(f"Perfil de Temperatura Transversal (Línea Central Y={centro_y})")
        ax_p.set_xlabel("Posición Horizontal (Píxeles)")
        ax_p.set_ylabel("Intensidad Térmica")
        ax_p.legend()
        ax_p.grid(True, alpha=0.3)
        st.pyplot(fig_perfil)

        # --- SECCIÓN 3: CLASIFICACIÓN CON MODELO DE IA Y BARRAS DE PROBABILIDAD ---
        st.markdown("---")
        st.subheader("🤖 Clasificación por Red Neuronal (CNN)")

        img_resized = cv2.resize(imagen_np, (64, 64))
        img_array = np.expand_dims(img_resized, axis=0)

        prediccion = modelo.predict(img_array)[0] # Arreglo de 3 probabilidades
        indice = np.argmax(prediccion)
        categoria_predicha = clases[indice]
        confianza = prediccion[indice] * 100

        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            if categoria_predicha == 'Enfriamiento Deficiente':
                st.error(f"⚠️ **Resultado:** {categoria_predicha}\n\n**Confianza:** {confianza:.1f}%")
            elif categoria_predicha == 'Enfriamiento Óptimo':
                st.success(f"✅ **Resultado:** {categoria_predicha}\n\n**Confianza:** {confianza:.1f}%")
            else:
                st.warning(f"❄️ **Resultado:** {categoria_predicha}\n\n**Confianza:** {confianza:.1f}%")

        with col_res2:
            # Gráfico de Barras de Distribución de Probabilidad
            fig_prob, ax_pb = plt.subplots(figsize=(6, 3))
            colores = ['#ff4b4b' if c == 'Enfriamiento Deficiente' else '#28a745' if c == 'Enfriamiento Óptimo' else '#17a2b8' for c in clases]
            ax_pb.barh(clases, prediccion * 100, color=colores)
            ax_pb.set_xlim(0, 100)
            ax_pb.set_xlabel("Probabilidad (%)")
            ax_pb.set_title("Distribución de Certidumbre del Modelo")
            for i, v in enumerate(prediccion * 100):
                ax_pb.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
            st.pyplot(fig_prob)

        # --- SECCIÓN 4: DESEMPENO GENERAL EN COLAB ---
        st.markdown("---")
        st.subheader("📈 Rendimiento Histórico de Modelos (Entrenamiento)")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Red Neuronal CNN", "100.0%", "Modelo Activo")
        m_col2.metric("Random Forest (RF)", "94.4%", "Machine Learning")
        m_col3.metric("CRNN (CNN + Recurrente)", "100.0%", "Modelo Híbrido")

        # Curva de precisión
        fig_acc, ax_acc = plt.subplots(figsize=(10, 3))
        epocas = np.arange(1, 11)
        acc_train = [0.45, 0.65, 0.80, 0.88, 0.92, 0.95, 0.98, 0.99, 1.0, 1.0]
        acc_val = [0.40, 0.60, 0.75, 0.85, 0.90, 0.95, 0.97, 1.0, 1.0, 1.0]
        
        ax_acc.plot(epocas, acc_train, label='Precisión Entrenamiento', marker='o')
        ax_acc.plot(epocas, acc_val, label='Precisión Validación', marker='s')
        ax_acc.set_title('Evolución de la Precisión durante el Entrenamiento')
        ax_acc.set_xlabel('Épocas')
        ax_acc.set_ylabel('Precisión')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        st.pyplot(fig_acc)

else:
    st.info("Sube una imagen térmica desde la barra lateral izquierda para iniciar el análisis.")
