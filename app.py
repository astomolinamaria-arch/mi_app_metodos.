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

# Configuración principal de la interfaz (con ícono personalizado para la app móvil)
st.set_page_config(
    page_title="Control Térmico Mango",
    page_icon="🥭",
    layout="wide"
)

st.title("Clasificación Termográfica en Néctar de Mango")
st.write("Aplicación interactiva para el análisis de gradiente térmico en la etapa de enfriamiento post-pasteurización.")

# --- SECCIÓN 0: VIDEO TUTORIAL Y GUÍA (LECTURA EN BYTES PARA EVITAR PANTALLA NEGRA) ---
with st.expander("📺 Ver Guía e Instructivo de Uso de la Aplicación"):
    st.write("En el siguiente video se explica el funcionamiento del análisis de gradiente y clasificación con IA:")
    try:
        with open("guia-de-instruccion.mp4.mp4", "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except FileNotFoundError:
        st.warning("⚠️ No se encontró el video local. Verifica que el nombre en GitHub coincida exactamente con el código.")

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

    titulo_style = ParagraphStyle('Titulo', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1A2B4C'), spaceAfter=6)
    subtitulo_style = ParagraphStyle('Subtitulo', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#D97706'), spaceAfter=8)
    fecha_style = ParagraphStyle('Fecha', parent=styles['Normal'], fontSize=9, textColor=colors.gray, spaceAfter=15)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14)

    # Encabezado Corporativo Personalizado
    story.append(Paragraph("<b>SISTEMA DE CONTROL DE CALIDAD TÉRMICA</b>", titulo_style))
    story.append(Paragraph("<b>Néctar de Mango | Etapa de Enfriamiento Post-Pasteurización</b>", subtitulo_style))
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Informe Técnico Auditado</b> | Fecha de Generación: {fecha_actual}", fecha_style))
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
        ["Producto / Etapa Evaluada", "Néctar de Mango - Enfriamiento Post-Pasteurización"],
        ["
