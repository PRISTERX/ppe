import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# ─────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PPE Detector",
    page_icon="🦺",
    layout="wide"
)

# ─────────────────────────────────────────
# Estilos CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
        background-color: #0d0d0d;
        color: #f0f0f0;
    }

    .stApp {
        background: #0d0d0d;
    }

    h1, h2, h3 {
        font-family: 'Share Tech Mono', monospace;
        color: #f0f0f0;
    }

    .header-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #111 100%);
        border-left: 4px solid #f5a623;
        padding: 24px 32px;
        margin-bottom: 32px;
        border-radius: 4px;
    }

    .header-box h1 {
        font-size: 2.2rem;
        margin: 0;
        color: #f5a623;
        letter-spacing: 2px;
    }

    .header-box p {
        margin: 6px 0 0;
        color: #888;
        font-size: 0.95rem;
        font-family: 'Share Tech Mono', monospace;
    }

    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #f5a623;
        padding: 20px;
        border-radius: 4px;
        text-align: center;
    }

    .metric-card .value {
        font-size: 2rem;
        font-family: 'Share Tech Mono', monospace;
        color: #f5a623;
        font-weight: bold;
    }

    .metric-card .label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .detection-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.9rem;
    }

    .conf-bar-container {
        background: #2a2a2a;
        border-radius: 2px;
        height: 6px;
        width: 100px;
        display: inline-block;
        vertical-align: middle;
        margin-left: 8px;
    }

    .stFileUploader > div {
        background: #1a1a1a !important;
        border: 2px dashed #333 !important;
        border-radius: 4px !important;
    }

    .stFileUploader > div:hover {
        border-color: #f5a623 !important;
    }

    .stSlider > div > div {
        color: #f5a623 !important;
    }

    div[data-testid="stSidebar"] {
        background: #111 !important;
        border-right: 1px solid #222;
    }

    .stButton > button {
        background: #f5a623;
        color: #000;
        font-family: 'Share Tech Mono', monospace;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 2px;
        letter-spacing: 1px;
        width: 100%;
    }

    .stButton > button:hover {
        background: #e09510;
        color: #000;
    }

    .alert-danger {
        background: #2a1a1a;
        border-left: 4px solid #ff4444;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        color: #ff4444;
        margin: 8px 0;
    }

    .alert-ok {
        background: #1a2a1a;
        border-left: 4px solid #44ff88;
        padding: 12px 16px;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        color: #44ff88;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🦺 PPE DETECTOR</h1>
    <p>YOLOv8n · Detección de Equipos de Protección Personal</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    model_path = st.text_input(
        "Ruta del modelo (best.pt)",
        value="/content/drive/MyDrive/ppe/ejecuciones/yolov8n-4/weights/best.pt",
        help="Ruta al archivo best.pt generado por el entrenamiento"
    )

    confidence = st.slider(
        "Confianza mínima",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Detecciones por debajo de este umbral serán ignoradas"
    )

    iou_thresh = st.slider(
        "IoU threshold (NMS)",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05
    )

    st.markdown("---")
    st.markdown("### 🏷️ Clases PPE")
    clases = ['botas', 'audífonos', 'gafas', 'guantes', 'casco', 'persona', 'chaleco']
    for c in clases:
        st.markdown(f"- `{c}`")

# ─────────────────────────────────────────
# Cargar modelo
# ─────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ─────────────────────────────────────────
# Colores por clase
# ─────────────────────────────────────────
CLASS_COLORS = {
    'botas.':    (255, 165,   0),
    'audifonos': (138,  43, 226),
    'gafas':     ( 30, 144, 255),
    'guantes':   (255,  20, 147),
    'casco':     (255, 215,   0),
    'persona':   ( 50, 205,  50),
    'chaleco':   (255,  69,   0),
}

# ─────────────────────────────────────────
# Upload de imagen
# ─────────────────────────────────────────
st.markdown("### 📁 Cargar imagen")
uploaded_file = st.file_uploader(
    "Arrastra o selecciona una imagen",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded_file:
    # Verificar modelo
    if not os.path.exists(model_path):
        st.error(f"❌ Modelo no encontrado en: `{model_path}`\n\nVerifica la ruta en el panel lateral.")
        st.stop()

    # Cargar modelo
    with st.spinner("Cargando modelo..."):
        model = load_model(model_path)

    # Procesar imagen
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Imagen original**")
        st.image(image, use_container_width=True)

    # Inferencia
    with st.spinner("Detectando PPE..."):
        results = model.predict(
            img_array,
            conf=confidence,
            iou=iou_thresh,
            verbose=False
        )

    result = results[0]
    boxes = result.boxes
    names = model.names

    # Dibujar detecciones
    img_drawn = img_array.copy()

    detecciones = []
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf_val:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_drawn, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_drawn, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        detecciones.append((cls_name, conf_val))

    with col2:
        st.markdown("**Detecciones**")
        st.image(img_drawn, use_container_width=True)

    # ── Métricas ──────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Resultados")

    m1, m2, m3, m4 = st.columns(4)

    clases_detectadas = list(set([d[0] for d in detecciones]))
    conf_promedio = np.mean([d[1] for d in detecciones]) if detecciones else 0
    ppe_critico = ['casco', 'chaleco', 'gafas', 'guantes']
    ppe_presente = [c for c in ppe_critico if c in clases_detectadas]

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(detecciones)}</div>
            <div class="label">Detecciones totales</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(clases_detectadas)}</div>
            <div class="label">Clases únicas</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{conf_promedio:.0%}</div>
            <div class="label">Confianza promedio</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(ppe_presente)}/{len(ppe_critico)}</div>
            <div class="label">PPE crítico detectado</div>
        </div>""", unsafe_allow_html=True)

    # ── Alerta de cumplimiento ─────────────
    faltantes = [c for c in ppe_critico if c not in clases_detectadas]
    if faltantes:
        st.markdown(f"""
        <div class="alert-danger">
            ⚠️ PPE FALTANTE: {', '.join(faltantes).upper()}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-ok">
            ✅ CUMPLIMIENTO COMPLETO — Todo el PPE crítico detectado
        </div>""", unsafe_allow_html=True)

    # ── Lista de detecciones ───────────────
    if detecciones:
        st.markdown("### 🔍 Detalle de detecciones")
        for cls_name, conf_val in sorted(detecciones, key=lambda x: -x[1]):
            bar_width = int(conf_val * 100)
            color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS.get(cls_name, (200,200,200)))
            st.markdown(f"""
            <div class="detection-row">
                <span style="color:{color_hex}">■</span>&nbsp;
                <span style="flex:1; margin-left:8px">{cls_name}</span>
                <span style="color:#f5a623; font-weight:bold">{conf_val:.1%}</span>
                <div class="conf-bar-container">
                    <div style="width:{bar_width}%; height:100%; background:{color_hex}; border-radius:2px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("No se detectaron objetos con la confianza mínima configurada. Intenta bajar el umbral.")

else:
    st.info("👆 Sube una imagen para comenzar la detección.")
