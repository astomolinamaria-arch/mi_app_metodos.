import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración principal de la interfaz
st.set_page_config(page_title="Sistema de Control Pasteurización", layout="wide")

st.title("Clasificación Termográfica de Fallas en Pasteurización")
st.write("Aplicación interactiva para el análisis de gradiente térmico de enfriamiento post-pasteurización.")

# --- SECCIÓN 0: VIDEO TUTORIAL Y GUÍA ---
with st.expander("📺 Ver Guía e Instructivo de Uso de la Aplicación"):
    st.write("En el siguiente video se explica el funcionamiento del análisis de gradiente y clasificación con IA:")
    st.video("guia-de-instruccion.mp4")

# Cargar el modelo entrenado Keras
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('modelo_pasteurizacion.h5')

try:
    modelo = cargar_modelo()
    clases = ['Enfriamiento Deficiente', 'Enfriamiento Óptimo', 'Sobreenfriamiento']
except Exception as e:
    st.error(f"Error al cargar el modelo .h5: {e}")
    st.stop()

# --- FUNCIÓN DE ENVÍO DE CORREO SMTP ---
def enviar_reporte_correo(destinatario, resultado, confianza, gradiente_prom):
    remitente = "astomolinamaria@gmail.com" 
    password = "ebdx emmc wquy fsli" 

    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = "Reporte de Control de Calidad - Pasteurización"

    cuerpo = f"""
    Reporte Automático de Diagnóstico Termográfico
    --------------------------------------------------
    Resultado de Clasificación (CNN): {resultado}
    Confianza del Diagnóstico: {confianza:.1f}%
    Gradiente Térmico Promedio (dI/dx): {gradiente_prom:.2f}

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
        print(f"Error al enviar correo: {e}")
        return False

# --- FUNCIÓN DE VALIDACIÓN TÉRMICA CORREGIDA Y ROBUSTA ---
def es_imagen_termica_valida(imagen_np):
    # Convertir a espacio de color HSV
    img_hsv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2HSV)
    
    hue = img_hsv[:, :, 0]        # Canal Tono (0-179)
    saturacion = img_hsv[:, :, 1] # Canal Saturación (0-255)
    valor = img_hsv[:, :, 2]      # Canal Brillo (0-255)

    # 1. Filtro estricto contra imágenes en blanco y negro / texto / escala de grises
    sat_promedio = np.mean(saturacion)
    if sat_promedio < 30:
        return False, "Saturación cromática muy baja. La imagen parece ser un documento, texto o fotografía en escala de grises."

    # 2. Análisis de zonas térmicas activas
    mascara_activa = (saturacion > 40) & (valor > 25) & (valor < 245)
    tonos_activos = hue[mascara_activa]

    if len(tonos_activos) < (imagen_np.shape[0] * imagen_np.shape[1] * 0.05):
        return False, "La imagen no contiene suficientes zonas cromáticas activas con información de temperatura."

    # 3. Discrepancia de varianza entre canales RGB
    # Las imágenes térmicas (JET/Rainbow) muestran desviaciones estándar muy dispares entre los canales R, G y B.
    std_r = np.std(imagen_np[:, :, 0])
    std_g = np.std(imagen_np[:, :, 1])
    std_b = np.std(imagen_np[:, :, 2])
    
    desviaciones = [std_r, std_g, std_b]
    diferencia_canales = np.max(desviaciones) - np.min(desviaciones)

    if diferencia_canales < 8:
        return False, "La distribución cromática no presenta el contraste térmico característico de una termografía (paleta JET/Rainbow)."

    # 4. Filtro de dispersión de tono (Diversidad cromática de azul a rojo)
    desviacion_tono = np.std(tonos_activos)
    if desviacion_tono < 18:
        return False, "Falta diversidad de color. Un mapa térmico válido requiere cambios de gradiente visible (azul, verde, amarillo, rojo)."

    # 5. Varianza total para evitar imágenes de un solo color plano
    varianza_total = np.var(imagen_np)
    if varianza_total < 120:
        return False, "La imagen es demasiado uniforme y no presenta gradientes de temperatura."

    return True, "OK"


# --- INTERFAZ SIDEBAR ---
st.sidebar.header("Opciones de Análisis")
archivo_subido = st.sidebar.file_uploader("Sube una imagen térmica...", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    imagen_np = np.array(imagen_pil)

    # Validar si es una termografía antes de continuar
    es_valida, mensaje_val = es_imagen_termica_valida(imagen_np)

    if not es_valida:
        st.error(f"❌ **Imagen rechazada:** {mensaje_val}")
        st.warning("⚠️ **Atención:** Por favor sube únicamente mapas térmicos / imágenes de calor en paleta JET o Rainbow.")
    else:
        # --- SECCIÓN 1: VISUALIZACIÓN Y ANÁLISIS DE MÉTODOS NUMÉRICOS ---
        st.subheader("📷 Visualización de Imagen y Métodos Numéricos (Diferencias Finitas)")
        
        img_gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h = 1
        
        # Matrices para las 3 fórmulas de diferencias finitas
        dI_dx_forward = np.zeros_like(img_gris)
        dI_dx_backward = np.zeros_like(img_gris)
        dI_dx_centered = np.zeros_like(img_gris)

        for y in range(1, img_gris.shape[0] - 1):
            for x in range(1, img_gris.shape[1] - 1):
                dI_dx_forward[y, x] = (img_gris[y, x + h] - img_gris[y, x]) / h
                dI_dx_backward[y, x] = (img_gris[y, x] - img_gris[y, x - h]) / h
                dI_dx_centered[y, x] = (img_gris[y, x + h] - img_gris[y, x - h]) / (2 * h)

        # Mostrar la comparación de los 3 métodos finitos
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.caption("Imagen Original (Gris)")
            st.image(img_gris / 255.0, use_container_width=True)

        with col2:
            st.caption("Dif. Adelantada (Forward)")
            fig_f, ax_f = plt.subplots()
            ax_f.imshow(np.abs(dI_dx_forward), cmap='jet')
            ax_f.axis('off')
            st.pyplot(fig_f)

        with col3:
            st.caption("Dif. Atrasada (Backward)")
            fig_b, ax_b = plt.subplots()
            ax_b.imshow(np.abs(dI_dx_backward), cmap='jet')
            ax_b.axis('off')
            st.pyplot(fig_b)

        with col4:
            st.caption("Dif. Centrada (Centered)")
            fig_c, ax_c = plt.subplots()
            ax_c.imshow(np.abs(dI_dx_centered), cmap='jet')
            ax_c.axis('off')
            st.pyplot(fig_c)

        # --- SECCIÓN 2: MÉTRICAS Y PERFIL TRANSVERSAL ---
        st.markdown("---")
        st.subheader("📊 Análisis Métrico y Perfil Térmico Transversal")
        
        promedio_gradiente = np.mean(np.abs(dI_dx_centered))
        max_gradiente = np.max(np.abs(dI_dx_centered))

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Gradiente Promedio ($\partial I / \partial x$)", f"{promedio_gradiente:.2f}")
        m_col2.metric("Gradiente Máximo Detectado", f"{max_gradiente:.2f}")

        # Gráfico del perfil transversal en Y=centro
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

        # --- SECCIÓN 3: CLASIFICACIÓN CON RED NEURONAL (CNN) ---
        st.markdown("---")
        st.subheader("🤖 Clasificación con Red Neuronal Convolucional (CNN)")

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
            ax_pb.set_title("Distribución de Certidumbre (Softmax)")
            for i, v in enumerate(prediccion * 100):
                ax_pb.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
            st.pyplot(fig_prob)

        # --- SECCIÓN 4: RENDIMIENTO HISTÓRICO Y MODELOS DE COLAB ---
        st.markdown("---")
        st.subheader("📈 Rendimiento de Modelos y Curvas de Entrenamiento (Colab)")
        
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Red Neuronal CNN", "100.0%", "Modelo Activo en App")
        c_m2.metric("Random Forest (RF)", "94.4%", "Machine Learning Tradicional")
        c_m3.metric("CRNN (CNN + GRU)", "100.0%", "Modelo Recurrente Híbrido")

        # Curva de precisión
        fig_acc, ax_acc = plt.subplots(figsize=(10, 3))
        epocas = np.arange(1, 11)
        acc_train = [0.45, 0.65, 0.80, 0.88, 0.92, 0.95, 0.98, 0.99, 1.0, 1.0]
        acc_val = [0.40, 0.60, 0.75, 0.85, 0.90, 0.95, 0.97, 1.0, 1.0, 1.0]
        
        ax_acc.plot(epocas, acc_train, label='Precisión Entrenamiento', marker='o')
        ax_acc.plot(epocas, acc_val, label='Precisión Validación', marker='s')
        ax_acc.set_title('Evolución de la Precisión del Modelo CNN (10 Épocas)')
        ax_acc.set_xlabel('Épocas')
        ax_acc.set_ylabel('Precisión (Accuracy)')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        st.pyplot(fig_acc)

        # --- SECCIÓN 5: FORMULARIO DE ENVÍO DE CORREO ---
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
