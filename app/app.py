import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image

# ==========================================
# 1. DEFINISI CLASS & FUNGSI
# ==========================================

class ManualBackprop:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return self.softmax(Z2)

def extract_hue_histogram(image, bins=32):
    # Konversi dari RGB (Streamlit/PIL) ke BGR (OpenCV)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

# ==========================================
# 2. KONFIGURASI & LOAD MODEL
# ==========================================

CATEGORIES = ['1000', '2000', '5000', '10000', '20000', '50000', '100000']

@st.cache_resource
def load_model():
    try:
        with open('backprop_rupiah_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Uji Klasifikasi", "Metodologi"])

# ==========================================
# HALAMAN 1: BERANDA
# ==========================================
if page == "Beranda":
    st.title("💰 Klasifikasi Uang Rupiah Kertas")
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Deskripsi Proyek")
        st.write("""
        Aplikasi ini merupakan implementasi Jaringan Syaraf Tiruan Backpropagation 
        yang dibangun secara manual menggunakan library NumPy. Sistem ini dirancang untuk 
        mengenali nominal uang kertas Rupiah berdasarkan ekstraksi ciri warna (HSV Histogram).
        """)
    with col2:
        st.metric("Akurasi Model", "92.00%")
    
    st.info("**Judul Tugas:** Implementasi Arsitektur Artificial Neural Network (ANN) Backpropagation Berbasis Ekstraksi Ciri Histogram HSV untuk Klasifikasi Multi-Kelas Uang Kertas Rupiah")
    
    st.write("---")
    st.subheader("Profil Pengembang")
    st.write("- **Nama:** Nabilah Shamid")
    st.write("- **NIM:** 09021382328147")
    st.write("- **Dosen Pengampu:** Assoc, Prof. Julian Supardi, S.Pd., M.T., Ph.D.")
    st.write("- **Mata Kuliah:** Jaringan Syaraf Tiruan")

# ==========================================
# HALAMAN 2: UJI KLASIFIKASI
# ==========================================
elif page == "Uji Klasifikasi":
    st.title("📸 Ruang Uji")
    
    if model is None:
        st.error("File 'backprop_rupiah_model.pkl' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    else:
        tab1, tab2 = st.tabs(["Ambil Kamera", "Unggah Gambar"])
        
        img_file = None
        with tab1:
            img_camera = st.camera_input("Jepret Uang Rupiah Kertas")
            if img_camera: img_file = img_camera
        with tab2:
            img_upload = st.file_uploader("Pilih file gambar...", type=["jpg", "png", "jpeg"])
            if img_upload: img_file = img_upload

        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Gambar Input", use_container_width=True)
            
            if st.button("Mulai Klasifikasi"):
                with st.spinner('Ekstraksi ciri dan prediksi sedang berlangsung...'):
                    # 1. Ekstraksi
                    features = extract_hue_histogram(image)
                    
                    # 2. Prediksi
                    pred_probs = model.predict(features.reshape(1, -1))[0]
                    idx = np.argmax(pred_probs)
                    result = CATEGORIES[idx]
                    confidence = pred_probs[idx] * 100
                    
                    # 3. Tampilkan Hasil
                    st.write("---")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(f"### Prediksi: Rp {result}")
                        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                    
                    with c2:
                        # Visualisasi Histogram Real-time
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(features, color='purple')
                        ax.fill_between(range(len(features)), features, color='purple', alpha=0.3)
                        ax.set_title("Fitur Histogram Hue")
                        st.pyplot(fig)

# ==========================================
# HALAMAN 3: METODOLOGI
# ==========================================
elif page == "Metodologi":
    st.title("🧠 Di Balik Layar")
    st.write("Bagian ini menjelaskan detail teknis dari arsitektur ANN yang dibangun.")
    
    st.subheader("Arsitektur Jaringan")
    st.code("""
    - Input Layer  : 32 Neuron (Histogram Bins)
    - Hidden Layer : 64 Neuron (Aktivasi ReLU)
    - Output Layer : 7 Neuron (Aktivasi Softmax)
    - Optimizer    : Manual Backpropagation
    - Loss Function: Cross Entropy
    """)
    
    st.subheader("Evaluasi Performa")
    st.write("Berdasarkan pengujian pada dataset 'Rupiah Banknotes', model mencapai akurasi 92% pada data uji.")
    
    st.warning("Confusion Matrix")
    st.image("confussion_matrix_backprop.png")
