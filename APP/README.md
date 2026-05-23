# README - Klasifikasi Mata Uang Rupiah

## Deskripsi Project
Project ini merupakan aplikasi klasifikasi mata uang rupiah kertas berbasis Backpropagation. Sistem mampu mengenali nominal uang rupiah menggunakan input kamera secara realtime.

Metode yang digunakan:
- Ekstraksi ciri menggunakan MobileNetV2
- Klasifikasi menggunakan Manual Backpropagation Neural Network

Aplikasi dikembangkan menggunakan:
- Python
- TensorFlow / Keras
- OpenCV

---

# Anggota Kelompok

1. Surya Andika Tandranparang (09021382328169) 
2. Nabilah Shamid (09021382328147)
3. Afny Chiara Wildani Nst (09021382328167)  

---

# Fitur Aplikasi

- Deteksi nominal uang rupiah realtime
- Menggunakan kamera/webcam
- Menampilkan confidence score prediksi
- Prediksi realtime menggunakan OpenCV
- Dapat dijalankan dalam bentuk executable (.exe)

---

# Dataset

Dataset yang digunakan berupa citra mata uang rupiah dengan beberapa kategori nominal:

- Rp1.000
- Rp2.000
- Rp5.000
- Rp10.000
- Rp20.000
- Rp50.000
- Rp100.000

---

# Metode yang Digunakan

## 1. Ekstraksi Ciri MobileNetV2

MobileNetV2 digunakan sebagai feature extractor untuk mengambil representasi fitur dari gambar uang rupiah.

Karakteristik:
- Input gambar: 224 × 224
- Output fitur: 1280 dimensi
- Menggunakan pretrained ImageNet
- Fully Connected bawaan dihapus (`include_top=False`)

---

## 2. Manual Backpropagation

Hasil ekstraksi fitur MobileNetV2 kemudian diproses menggunakan jaringan saraf tiruan manual berbasis backpropagation.

Arsitektur:
- Input Layer : 1280 neuron
- Hidden Layer : 2560 neuron
- Output Layer : 7 neuron

Activation Function:
- ReLU
- Softmax

Loss Function:
- Cross Entropy

---

# Cara Menjalankan Aplikasi

## Menggunakan File Installer / Executable

1. Jalankan file:
```text
app_kamera.exe
```

2. Tunggu hingga kamera aktif.

3. Arahkan uang rupiah ke kamera.

4. Sistem akan menampilkan:
- nominal uang,
- confidence prediksi.

---

# Cara Menggunakan Kamera

- Pastikan webcam aktif.
- Tempatkan uang rupiah pada area yang cukup terang.
- Arahkan uang ke depan kamera.
- Tunggu hasil prediksi muncul pada layar.

---

# Cara Menutup Aplikasi

Untuk menutup aplikasi:

```text
Tekan tombol Q pada keyboard
```

Setelah tombol `Q` ditekan:
- kamera akan berhenti,
- window aplikasi tertutup otomatis.

---

# Dependency

Library utama yang digunakan:
- TensorFlow
- Keras
- OpenCV
- NumPy

---

# Struktur Project

```text
rupiah_app/
│
├── app_kamera.py
├── labels.json
├── requirements.txt
├── rupiah_classifier_mobilenet_full.h5
├── mobilenet_v2_weights.h5
└── app_kamera.exe
```

---

# Catatan

- Aplikasi membutuhkan webcam/kamera.
- Startup pertama mungkin sedikit lambat karena proses load model.
- Pencahayaan yang baik akan meningkatkan akurasi prediksi.

---

# Teknologi yang Digunakan

- Python
- TensorFlow
- Keras
- OpenCV
- MobileNetV2
- Deep Learning
- Computer Vision
