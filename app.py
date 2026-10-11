import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración de la página
st.set_page_config(page_title="Clasificación Termográfica", layout="wide")

st.title("Clasificación Termográfica de Fallas en Pasteurización")
st.write("Aplicación interactiva para el análisis de gradiente térmico de enfriamiento post-pasteurización.")

# --- SECCIÓN 0: VIDEO TUTORIAL Y GUÍA ---
with st.expander("📺 Ver Guía e Instructivo de Uso de la Aplicación"):
    st.write("A continuación se muestra el video explicativo sobre el uso y análisis del sistema:")
    st.video("https://www.youtube.com/watch?v=TU_VIDEO_AQUI")

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

# --- FUNCIÓN PARA ENVIAR CORREO ---
def enviar_reporte_correo(destinatario, resultado, confianza, gradiente_prom):
    # ⚠️ REEMPLAZA ESTOS DOS VALORES CON TUS DATOS REALES:
    remitente = "astomolinamaria@gmail.com" 
    password = "ebdx emmc wquy fsli"  # Tu contraseña de 16 letras de Google

    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = "Reporte de Análisis Termográfico - Pasteurización"

    cuerpo = f"""
    Reporte Automático de Control de Calidad
    -------------------------------------------
    Resultado de Clasificación: {resultado}
    Nivel de Confianza: {confianza:.1f}%
    Gradiente Térmico Promedio: {gradiente_prom:.2f}

    Este reporte fue generado desde la aplicación web de Pasteurización.
    """
    msg.attach(MIMEText(cuerpo, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(remitente, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# --- FUNCIÓN DE VALIDACIÓN DE IMAGEN TÉRMICA ---
def es_imagen_termica_valida(imagen_np):
    img_hsv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2HSV)
    saturacion_promedio = np.mean(img_hsv[:, :, 1])
    desviacion_rgb = np.std(imagen_np, axis=(0, 1))
    
    if saturacion_promedio < 35 or np.mean(desviacion_rgb) < 20:
        return False, "La imagen no presenta la saturación cromática de un mapa térmico válido."
    return True, "OK"

# Menú lateral
st.sidebar.header("Opciones de Análisis")
archivo_subido = st.sidebar.file_uploader("Sube una imagen térmica...", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    imagen_np = np.array(imagen_pil)

    es_valida, mensaje_val = es_imagen_termica_valida(imagen_np)

    if not es_valida:
        st.error(f"❌ **Imagen rechazada:** {mensaje_val}")
        st.warning("⚠️ **Atención:** Por favor sube una imagen térmica válida.")
    else:
        # SECCIÓN 1: VISUALIZACIÓN Y MÉTODOS NUMÉRICOS
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Imagen Original")
            st.image(imagen_pil, use_container_width=True)

        with col2:
            st.subheader("🔥 Gradiente Térmico (Diferencia Centrada)")
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

        promedio_gradiente = np.mean(np.abs(dI_dx_centered))
        max_gradiente = np.max(np.abs(dI_dx_centered))

        st.markdown("---")
        st.subheader("📊 Análisis Métrico de Gradiente")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Gradiente Promedio ($\partial I / \partial x$)", f"{promedio_gradiente:.2f}")
        col_m2.metric("Gradiente Máximo Detectado", f"{max_gradiente:.2f}")

        # SECCIÓN 2: CLASIFICACIÓN CON IA
        st.markdown("---")
        st.subheader("🤖 Clasificación por Red Neuronal (CNN)")

        img_resized = cv2.resize(imagen_np, (64, 64))
        img_array = np.expand_dims(img_resized, axis=0)

        prediccion = modelo.predict(img_array)[0]
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
            fig_prob, ax_pb = plt.subplots(figsize=(6, 3))
            colores = ['#ff4b4b' if c == 'Enfriamiento Deficiente' else '#28a745' if c == 'Enfriamiento Óptimo' else '#17a2b8' for c in clases]
            ax_pb.barh(clases, prediccion * 100, color=colores)
            ax_pb.set_xlim(0, 100)
            ax_pb.set_xlabel("Probabilidad (%)")
            ax_pb.set_title("Distribución de Certidumbre")
            for i, v in enumerate(prediccion * 100):
                ax_pb.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
            st.pyplot(fig_prob)

        # SECCIÓN 3: FORMULARIO DE ENVÍO DE CORREO
        st.markdown("---")
        st.subheader("📧 Enviar Reporte por Correo Electrónico")

        with st.form("form_correo"):
            correo_destino = st.text_input("Ingrese el correo electrónico del destinatario:")
            boton_enviar = st.form_submit_button("Enviar Reporte")

            if boton_enviar:
                if correo_destino:
                    exito = enviar_reporte_correo(correo_destino, categoria_predicha, confianza, promedio_gradiente)
                    if exito:
                        st.success(f"✅ Reporte enviado exitosamente a {correo_destino}")
                    else:
                        st.error("❌ Ocurrió un error al enviar el correo. Revisa las credenciales SMTP.")
                else:
                    st.warning("Por favor ingrese un correo válido.")

else:
    st.info("Sube una imagen térmica desde la barra lateral izquierda para iniciar el análisis.")
