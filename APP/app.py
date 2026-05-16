import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Klasifikasi Uang Rupiah",
    page_icon="💰",
    layout="centered",
)

# ==========================================
# KONSTANTA
# ==========================================

CATEGORIES = ['1000', '2000', '5000', '10000', '20000', '50000', '100000']
IMG_SIZE   = (224, 224)
MODEL_DIR  = "rupiah_classifier_mobilenet.keras"   # folder .keras

# ==========================================
# LOAD MODEL (cached agar tidak reload tiap interaksi)
# ==========================================

@st.cache_resource
def load_models():
    # Feature extractor MobileNetV2 (frozen, pretrained ImageNet)
    mobilenet = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling="avg",
        weights="imagenet",
    )
    mobilenet.trainable = False

    # Classifier (Dense) yang sudah dilatih
    classifier = tf.keras.models.load_model(MODEL_DIR)

    return mobilenet, classifier


# ==========================================
# FUNGSI EKSTRAKSI CIRI & PREDIKSI
# ==========================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Konversi PIL Image → array siap MobileNetV2."""
    img_rgb     = image.convert("RGB")
    img_resized = img_rgb.resize(IMG_SIZE)
    img_array   = np.array(img_resized, dtype="float32")
    img_array   = np.expand_dims(img_array, axis=0)          # (1, 224, 224, 3)
    return preprocess_input(img_array)                        # normalisasi ke [-1, 1]


def classify(image: Image.Image, mobilenet, classifier):
    """Jalankan pipeline: ekstraksi ciri MobileNetV2 → prediksi classifier."""
    img_preprocessed = preprocess_image(image)
    features         = mobilenet.predict(img_preprocessed, verbose=0)  # (1, 1280)
    probs            = classifier.predict(features, verbose=0)[0]       # (7,)
    idx              = int(np.argmax(probs))
    return CATEGORIES[idx], float(probs[idx]) * 100, probs


# ==========================================
# UI UTAMA
# ==========================================

st.markdown(
    "<h1 style='text-align:center;'>💰 Klasifikasi Uang Rupiah Kertas</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray;'>MobileNetV2 + Backpropagation · ANN</p>",
    unsafe_allow_html=True,
)
st.divider()

# --- Load model dengan spinner ---
with st.spinner("Memuat model..."):
    try:
        mobilenet, classifier = load_models()
        model_ready = True
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        model_ready = False

if model_ready:
    # --- Input kamera ---
    img_input = st.camera_input("📷 Arahkan kamera ke uang kertas Rupiah")

    if img_input is not None:
        image = Image.open(img_input)

        with st.spinner("Menganalisis gambar..."):
            label, confidence, probs = classify(image, mobilenet, classifier)

        st.divider()

        # --- Hasil ---
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.image(image, use_container_width=True, caption="Gambar yang diambil")

        with col_right:
            st.metric(label="Prediksi Nominal", value=f"Rp {label}")
            st.metric(label="Tingkat Keyakinan", value=f"{confidence:.1f}%")

            # Progress bar keyakinan
            color = "green" if confidence >= 75 else "orange" if confidence >= 50 else "red"
            st.markdown(
                f"""
                <div style='margin-top:8px; background:#e0e0e0; border-radius:8px; height:14px; overflow:hidden;'>
                    <div style='width:{confidence:.0f}%; background:{color}; height:100%; border-radius:8px;'></div>
                </div>
                <p style='font-size:12px; color:gray; margin-top:4px;'>{confidence:.1f}% yakin</p>
                """,
                unsafe_allow_html=True,
            )

        # --- Distribusi probabilitas semua kelas ---
        st.divider()
        st.markdown("**Distribusi Probabilitas Semua Kelas**")

        prob_data = {f"Rp {cat}": float(p) * 100 for cat, p in zip(CATEGORIES, probs)}
        sorted_data = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True))

        for cat_label, prob_val in sorted_data.items():
            bar_color = "#4CAF50" if cat_label == f"Rp {label}" else "#90CAF9"
            st.markdown(
                f"""
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <span style='width:90px; font-size:13px;'>{cat_label}</span>
                    <div style='flex:1; background:#e0e0e0; border-radius:6px; height:12px; margin:0 10px; overflow:hidden;'>
                        <div style='width:{prob_val:.1f}%; background:{bar_color}; height:100%; border-radius:6px;'></div>
                    </div>
                    <span style='font-size:13px; width:50px; text-align:right;'>{prob_val:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
