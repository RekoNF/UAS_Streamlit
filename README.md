# 📚 Analisis Sentimen – Manga Evo

Aplikasi Streamlit untuk analisis sentimen ulasan aplikasi Manga Evo dari Google Play Store.

## 🚀 Cara Deploy

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi

```bash
streamlit run app.py
```

### 3. Deploy ke Streamlit Cloud (gratis)

1. Push repo ke GitHub
2. Buka https://share.streamlit.io
3. Hubungkan repo → pilih `app.py` → klik **Deploy**

---

## 📦 File

| File | Keterangan |
|------|-----------|
| `app.py` | Kode utama Streamlit |
| `requirements.txt` | Library yang diperlukan |

## 🔧 Fitur

- Scraping 100–1.000 ulasan langsung dari Google Play Store
- Text preprocessing (case folding, stopwords, lemmatisasi)
- Word Cloud & bar chart frekuensi kata
- Pelatihan Logistic Regression + TF-IDF
- Confusion matrix & classification report
- Prediksi sentimen teks baru secara real-time
- Export data ke CSV

---
*Tugas UAS · Eksplorasi dan Visualisasi Data · Statistika Universitas Matana 2025/2026*
