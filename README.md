# 🩺 Aplikasi Prediksi Diabetes dengan Range Glukosa Diperbarui

Aplikasi web untuk memprediksi risiko diabetes menggunakan Machine Learning dengan range glukosa yang realistis (hingga 400 mg/dL).

## 🚀 Fitur Utama

- ✅ **Range glukosa diperbarui**: 50-400 mg/dL (mencakup nilai diabetes)
- ✅ Visualisasi skala glukosa dengan kategori medis
- ✅ Prediksi risiko diabetes berdasarkan 8 parameter
- ✅ Analisis detail glukosa dengan kriteria ADA
- ✅ Riwayat prediksi dengan tracking glukosa
- ✅ Rekomendasi personalisasi berdasarkan level glukosa
- ✅ Feature importance visualization

## 🩺 **Kriteria Glukosa (American Diabetes Association)**

### **Diabetes:**
- Glukosa puasa ≥ 126 mg/dL
- Glukosa 2 jam ≥ 200 mg/dL
- HbA1c ≥ 6.5%

### **Prediabetes:**
- Glukosa puasa: 100-125 mg/dL
- Glukosa 2 jam: 140-199 mg/dL
- HbA1c: 5.7-6.4%

### **Normal:**
- Glukosa puasa < 100 mg/dL
- Glukosa 2 jam < 140 mg/dL
- HbA1c < 5.7%

## 📋 Parameter Input

1. **Kehamilan** (0-20)
2. **Glukosa** (50-400 mg/dL) ⭐ **Diperbarui!**
3. **Tekanan Darah** (40-180 mm Hg)
4. **Ketebalan Kulit** (0-99 mm)
5. **Insulin** (0-1000 µU/mL)
6. **BMI** (10-60 kg/m²)
7. **Riwayat Diabetes Keluarga** (0.08-2.50)
8. **Usia** (0-100 tahun)

## 🛠️ Teknologi

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **ML Model**: Random Forest Classifier
- **Accuracy**: ~77%
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib

## 📁 Struktur Project
