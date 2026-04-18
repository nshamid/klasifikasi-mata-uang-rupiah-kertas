# Implementasi Arsitektur Artificial Neural Network (ANN) Backpropagation Berbasis Ekstraksi Ciri Histogram HSV untuk Klasifikasi Multi-Kelas Uang Kertas Rupiah

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Manual_Implementation-013243.svg)](https://numpy.org/)

Proyek ini dikembangkan sebagai tugas **Ujian Tengah Semester (UTS)** mata kuliah Jaringan Syaraf Tiruan. Fokus utama proyek ini adalah membangun model klasifikasi uang kertas Rupiah (7 nominal) menggunakan arsitektur **ANN Backpropagation** yang diimplementasikan secara manual menggunakan library **NumPy**.

## 🚀 Fitur Utama
- **Manual JST Implementation**: Algoritma Backpropagation, fungsi aktivasi ReLU & Softmax, serta perhitungan Cross-Entropy dirancang dari nol tanpa library machine learning (seperti Scikit-Learn atau TensorFlow).
- **Ekstraksi Ciri HSV**: Menggunakan fitur Histogram Hue (32 bins) untuk mengenali sidik jari warna unik pada setiap pecahan uang.
- **Dashboard Interaktif**: Antarmuka berbasis Web menggunakan Streamlit yang mendukung prediksi melalui kamera (real-time) maupun unggah file gambar.
- **Visualisasi Hasil**: Dilengkapi dengan grafik histogram warna untuk setiap hasil prediksi guna transparansi proses ekstraksi ciri.

## 🧠 Arsitektur Jaringan
Sistem ini menggunakan spesifikasi teknis sebagai berikut:
- **Input Layer**: 32 Neuron (mewakili 32 bins dari Histogram Hue).
- **Hidden Layer**: 64 Neuron dengan fungsi aktivasi **ReLU**.
- **Output Layer**: 7 Neuron (nominal 1k hingga 100k) dengan fungsi aktivasi **Softmax**.
- **Metode Evaluasi**: Categorical Cross-Entropy Loss.
- **Akurasi Akhir**: Berhasil mencapai **92%** pada data uji.

## 📂 Struktur Repositori
```text
.
├── app/
│   ├── app.py                      # Skrip utama Dashboard Streamlit
│   └── requirements.txt            # Daftar library yang diperlukan
├── backprop_rupiah_model.pkl       # Model JST yang sudah dilatih
├── confussion_matrix_backprop.png  # Visualisasi performa model
├── klasifikasi-mata-uang-rupiah.ipynb # Dokumentasi proses training di Kaggle
└── README.md                       # Dokumentasi proyek
