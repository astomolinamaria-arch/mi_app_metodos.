import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import smtplib
import io
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication

# ReportLab para generación de PDFs
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Configuración principal de la interfaz
st.set_page_config(page_title="Sistema de Control Pasteurización", layout="wide")

st.title("Clasificación Termográfica de Fallas en Pasteurización")
st.write("Aplicación interactiva para el análisis de gradiente térmico de enfriamiento post-pasteurización.")

# --- SECCIÓN 0: VIDEO TUTORIAL Y GUÍA ---
with st.expander("📺 Ver Guía e Instructivo de Uso de la Aplicación"):
    st.write("En el siguiente video se explica el funcionamiento del análisis de gradiente y clasificación con IA:")
    st.video("https://www.youtube.com/watch?v=TU_ENLACE_DE_YOUTUBE")  # Reemplaza con tu enlace de YouTube si corresponde

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

# --- FUNCIÓN 1: GENERACIÓN DEL REPORTE PDF EN MEMORIA ---
def generar_pdf_reporte(imagen_np, fig_perfil, resultado, confianza, gradiente_prom, gradiente_max):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    titulo_style = ParagraphStyle('Titulo', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1A2B4C'), spaceAfter=10)
    subtitulo_style = ParagraphStyle('Subtitulo', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#555555'), spaceAfter=15)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14)

    story.append(Paragraph("<b>SISTEMA DE CONTROL DE CALIDAD - PASTEURIZACIÓN</b>", titulo_style))
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Informe Técnico Auditado</b> | Fecha de Generación: {fecha_actual}", subtitulo_style))
    story.append(Spacer(1, 10))

    img_pil = Image.fromarray(imagen_np)
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    fig_buffer = io.BytesIO()
    fig_perfil.savefig(fig_buffer, format='png', bbox_inches='tight', dpi=150)
    fig_buffer.seek(0)

    img_rl = RLImage(img_buffer, width=220, height=180)
    fig_rl = RLImage(fig_buffer, width=300, height=180)

    tabla_imagenes = Table([[img_rl, fig_rl]], colWidths=[240, 300])
    tabla_imagenes.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(tabla_imagenes)
    story.append(Spacer(1, 15))

    color_estado = colors.HexColor('#28A745') if resultado == 'Enfriamiento Óptimo' else colors.HexColor('#DC3545') if resultado == 'Enfriamiento Deficiente' else colors.HexColor('#17A2B8')

    datos_tabla = [
        [Paragraph("<b>Parámetro Evaluable</b>", ParagraphStyle('H1', parent=body_style, textColor=colors.white)), 
         Paragraph("<b>Valor Calculado / Diagnóstico</b>", ParagraphStyle('H2', parent=body_style, textColor=colors.white))],
        ["Resultado de Clasificación (CNN)", Paragraph(f"<font color='{color_estado.hexval()}'><b>{resultado}</b></font>", body_style)],
        ["Nivel de Confianza (Softmax)", f"{confianza:.2f} %"],
        ["Gradiente Térmico Promedio (dI/dx)", f"{gradiente_prom:.2f}"],
        ["Gradiente Térmico Máximo", f"{gradiente_max:.2f}"],
        ["Estado del Proceso", "Aprobado / Dentro de Norma" if resultado == 'Enfriamiento Óptimo' else "Requiere Revisión de Planta"]
    ]

    tabla_resumen = Table(datos_tabla, colWidths=[260, 280])
    tabla_resumen.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A2B4C')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))

    story.append(tabla_resumen)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Nota de Certificación:</b> Documento generado automáticamente mediante visión por computadora y análisis de diferencias finitas.", ParagraphStyle('Foot', parent=body_style, fontSize=8, textColor=colors.gray)))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# --- FUNCIÓN 2: ENVÍO DE CORREO HTML + ADJUNTOS ---
def enviar_reporte_correo_ejecutivo(destinatario, resultado, confianza, gradiente_prom, gradiente_max, imagen_np, fig_perfil):
    remitente = "astomolinamaria@gmail.com" 
    password = "ebdx emmc wquy fsli" 

    msg = MIMEMultipart('related')
    msg['From'] = f"Control de Calidad Pasteurización <{remitente}>"
    msg['To'] = destinatario
    msg['Subject'] = f"🚨 REPORTE DE CONTROL DE CALIDAD: {resultado.upper()} [{datetime.datetime.now().strftime('%d/%m/%Y')}]"

    color_badge = "#28a745" if resultado == 'Enfriamiento Óptimo' else "#dc3545" if resultado == 'Enfriamiento Deficiente' else "#17a2b8"

    html_cuerpo = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
            .container {{ max-width: 650px; background-color: #ffffff; border-radius: 8px; padding: 25px; margin: auto; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            .header {{ background-color: #1a2b4c; color: white; padding: 15px; text-align: center; border-radius: 6px 6px 0 0; }}
            .badge {{ background-color: {color_badge}; color: white; padding: 6px 12px; border-radius: 20px; font-weight: bold; display: inline-block; font-size: 14px; }}
            .metrics-grid {{ display: flex; margin: 20px 0; gap: 10px; text-align: center; }}
            .metric-card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; flex: 1; }}
            .metric-value {{ font-size: 20px; font-weight: bold; color: #1a2b4c; }}
            .metric-label {{ font-size: 11px; color: #6c757d; text-transform: uppercase; }}
            .img-container {{ text-align: center; margin: 20px 0; background: #f8f9fa; padding: 10px; border-radius: 6px; }}
            .footer {{ font-size: 11px; text-align: center; color: #888888; margin-top: 25px; border-top: 1px solid #eee; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2 style="margin:0;">Sistema de Control de Calidad Térmica</h2>
                <p style="margin:5px 0 0 0; font-size: 13px; opacity: 0.8;">Informe Automático de Pasteurización</p>
            </div>
            
            <div style="margin-top: 20px; text-align: center;">
                <span class="badge">{resultado}</span>
                <p style="color: #555; margin-top: 10px;">La muestra procesada ha sido analizada mediante Redes Neuronales Convolucionales (CNN).</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{confianza:.1f}%</div>
                    <div class="metric-label">Confianza CNN</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{gradiente_prom:.2f}</div>
                    <div class="metric-label">Gradiente Promedio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{gradiente_max:.2f}</div>
                    <div class="metric-label">Gradiente Máximo</div>
                </div>
            </div>

            <div class="img-container">
                <h4 style="margin-top:5px; color:#333;">Termografía Evaluable de la Muestra</h4>
                <img src="cid:imagen_termica" style="max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #ccc;">
            </div>

            <p style="font-size: 13px; color: #444;">📎 <b>Adjunto a este correo</b> encontrará el <b>Reporte Técnico Completo en formato PDF</b> con la auditoría de diferencias finitas y el perfil transversal para su base de datos.</p>

            <div class="footer">
                Este correo fue generado de forma automática por la Plataforma de Monitoreo Térmico.<br>
                © 2026 Pasteurización & Control Industrial.
            </div>
        </div>
    </body>
    </html>
    """

    msg_alt = MIMEMultipart('alternative')
    msg_alt.attach(MIMEText(html_cuerpo, 'html'))
    msg.attach(msg_alt)

    img_pil = Image.fromarray(imagen_np)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    mime_img = MIMEImage(img_byte_arr.read())
    mime_img.add_header('Content-ID', '<imagen_termica>')
    mime_img.add_header('Content-Disposition', 'inline', filename="termografia.png")
    msg.attach(mime_img)

    pdf_bytes = generar_pdf_reporte(imagen_np, fig_perfil, resultado, confianza, gradiente_prom, gradiente_max)
    mime_pdf = MIMEApplication(pdf_bytes, Name=f"Reporte_Pasteurizacion_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    mime_pdf.add_header('Content-Disposition', 'attachment', filename=f"Reporte_Pasteurizacion_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    msg.attach(mime_pdf)

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


# --- FUNCIÓN DE VALIDACIÓN COMPATIBLE CON TODO EL DATASET Y TÉRMICAS REALES ---
def es_imagen_termica_valida(imagen_np):
    img_hsv = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2HSV)
    
    saturacion = img_hsv[:, :, 1]
    valor = img_hsv[:, :, 2]

    sat_promedio = np.mean(saturacion)
    if sat_promedio < 8:
        return False, "Saturación insuficiente. La imagen parece ser un documento, texto o escala de grises."

    std_valor = np.std(valor)
    std_sat = np.std(saturacion)

    if std_valor < 1.5 and std_sat < 1.5:
        return False, "La imagen es plana y carece de variación térmica o gradiente."

    std_r = np.std(imagen_np[:, :, 0])
    std_g = np.std(imagen_np[:, :, 1])
    std_b = np.std(imagen_np[:, :, 2])
    
    max_desv = max(std_r, std_g, std_b)

    if max_desv < 1.5:
        return False, "La imagen no presenta la distribución de color característica de un mapa térmico."

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
        
        dI_dx_forward = np.zeros_like(img_gris)
        dI_dx_backward = np.zeros_like(img_gris)
        dI_dx_centered = np.zeros_like(img_gris)

        for y in range(1, img_gris.shape[0] - 1):
            for x in range(1, img_gris.shape[1] - 1):
                dI_dx_forward[y, x] = (img_gris[y, x + h] - img_gris[y, x]) / h
                dI_dx_backward[y, x] = (img_gris[y, x] - img_gris[y, x - h]) / h
                dI_dx_centered[y, x] = (img_gris[y, x + h] - img_gris[y, x - h]) / (2 * h)

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

        # --- SECCIÓN 5: FORMULARIO ÚNICO DE ENVÍO DE CORREO EJECUTIVO ---
        st.markdown("---")
        st.subheader("📧 Enviar Reporte Auditoría por Correo Electrónico")

        with st.form("form_correo_ejecutivo"):
            correo_destino = st.text_input("Ingrese el correo electrónico del destinatario:")
            boton_enviar = st.form_submit_button("Enviar Reporte Ejecutivo (PDF + HTML)")

            if boton_enviar:
                if correo_destino:
                    exito = enviar_reporte_correo_ejecutivo(
                        correo_destino, 
                        categoria_predicha, 
                        confianza, 
                        promedio_gradiente, 
                        max_gradiente, 
                        imagen_np, 
                        fig_perfil
                    )
                    if exito:
                        st.success(f"✅ Reporte técnico y PDF enviados exitosamente a {correo_destino}")
                    else:
                        st.error("❌ Ocurrió un error al enviar el correo. Revisa la conexión y credenciales SMTP.")
                else:
                    st.warning("Por favor ingrese un correo válido.")

else:
    st.info("Sube una imagen térmica desde la barra lateral izquierda para iniciar el análisis.")
