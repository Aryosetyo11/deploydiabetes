import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Titles */
    .main-title {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #2E86AB;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1a5c7c;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #2E86AB;
    }
    
    /* Success and Error boxes */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    /* Glukosa indicator */
    .glukosa-normal {
        color: #28a745;
        font-weight: bold;
    }
    
    .glukosa-prediabetes {
        color: #ffc107;
        font-weight: bold;
    }
    
    .glukosa-diabetes {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

# ============================================
# LOAD MODEL FUNCTION
# ============================================
def load_models():
    """
    Load machine learning model and scaler
    Using pickle instead of joblib for better compatibility
    """
    try:
        with open('best_diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {e}")
        st.info("Pastikan file 'best_diabetes_model.pkl' dan 'scaler.pkl' ada di folder yang sama")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ============================================
# USER INPUT FUNCTION - DENGAN RANGE GLUKOSA FIXED
# ============================================
def get_user_input():
    """
    Get user input from sidebar sliders
    Dengan range glukosa yang benar (hingga 400 mg/dL)
    """
    st.sidebar.header("📊 **Input Data Pasien**")
    st.sidebar.markdown("---")
    
    # Create two columns for better organization
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # 1. PREGNANCIES
        pregnancies = st.slider(
            '**Kehamilan**', 
            0, 20, 1,
            help="Jumlah kali hamil",
            key='pregnancies_input'
        )
        
        # 2. GLUCOSE - PERBAIKAN RANGE (50-400 mg/dL)
        glucose = st.slider(
            '**Glukosa** (mg/dL)', 
            50, 400, 120,
            help="""Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral

📊 **KATEGORI GLUKOSA (2 jam setelah makan):**
• 🟢 **Normal**: < 140 mg/dL
• 🟡 **Prediabetes**: 140-199 mg/dL  
• 🔴 **Diabetes**: ≥ 200 mg/dL

📌 **KATEGORI GLUKOSA (puasa 8 jam):**
• Normal: < 100 mg/dL
• Prediabetes: 100-125 mg/dL
• Diabetes: ≥ 126 mg/dL""",
            key='glucose_input'
        )
        
        # Tampilkan kategori glukosa berdasarkan nilai
        glucose_status = ""
        glucose_class = ""
        if glucose < 140:
            glucose_status = "🟢 Normal"
            glucose_class = "glukosa-normal"
        elif glucose < 200:
            glucose_status = "🟡 Prediabetes"
            glucose_class = "glukosa-prediabetes"
        else:
            glucose_status = "🔴 Diabetes"
            glucose_class = "glukosa-diabetes"
        
        st.sidebar.markdown(f"<div class='{glucose_class}'>Status Glukosa: {glucose_status}</div>", unsafe_allow_html=True)
        
        # 3. BLOOD PRESSURE
        blood_pressure = st.slider(
            '**Tekanan Darah** (mm Hg)', 
            40, 180, 70,
            help="Tekanan darah diastolik\n\n" +
                 "📊 **KATEGORI:**\n" +
                 "• Normal: < 80 mm Hg\n" +
                 "• Prehipertensi: 80-89 mm Hg\n" +
                 "• Hipertensi Stage 1: 90-99 mm Hg\n" +
                 "• Hipertensi Stage 2: ≥ 100 mm Hg",
            key='blood_pressure_input'
        )
        
        # 4. SKIN THICKNESS
        skin_thickness = st.slider(
            '**Ketebalan Kulit** (mm)', 
            0, 99, 20,
            help="Ketebalan lipatan kulit trisep (normal: 10-40 mm)",
            key='skin_thickness_input'
        )
    
    with col2:
        # 5. INSULIN
        insulin = st.slider(
            '**Insulin** (µU/mL)', 
            0, 1000, 80,
            help="Insulin serum 2 jam\n\n" +
                 "📊 **NILAI NORMAL:**\n" +
                 "• Puasa: 2-25 µU/mL\n" +
                 "• Setelah makan: < 100 µU/mL",
            key='insulin_input'
        )
        
        # 6. BMI
        bmi = st.slider(
            '**BMI** (kg/m²)', 
            10.0, 60.0, 25.0, 0.1,
            help="Indeks Massa Tubuh\n\n" +
                 "📊 **KATEGORI BMI:**\n" +
                 "• Kurus: < 18.5\n" +
                 "• Normal: 18.5-24.9\n" +
                 "• Gemuk: 25-29.9\n" +
                 "• Obesitas: ≥ 30",
            key='bmi_input'
        )
        
        # Tampilkan kategori BMI
        bmi_status = ""
        if bmi < 18.5:
            bmi_status = "⚪ Kurus"
        elif bmi < 25:
            bmi_status = "🟢 Normal"
        elif bmi < 30:
            bmi_status = "🟡 Gemuk"
        else:
            bmi_status = "🔴 Obesitas"
        
        st.sidebar.caption(f"Status BMI: {bmi_status}")
        
        # 7. DIABETES PEDIGREE FUNCTION
        diabetes_pedigree = st.slider(
            '**Riwayat Diabetes Keluarga**', 
            0.08, 2.50, 0.5, 0.01,
            help="""Fungsi silsilah diabetes (DPF)
            
📊 **INTERPRETASI:**
• < 0.5: Risiko rendah
• 0.5-1.0: Risiko sedang  
• > 1.0: Risiko tinggi""",
            key='diabetes_pedigree_input'
        )
        
        # 8. AGE
        age = st.slider(
            '**Usia** (tahun)', 
            0, 100, 30,
            help="Usia pasien",
            key='age_input'
        )
    
    st.sidebar.markdown("---")
    
    # Warning jika glukosa tinggi
    if glucose >= 200:
        st.sidebar.error("""
        ⚠️ **PERINGATAN: Nilai Glukosa Tinggi!**
        
        Glukosa ≥ 200 mg/dL termasuk kategori **DIABETES**.
        Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.
        """)
    
    # Create data dictionary
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    return pd.DataFrame(data, index=[0])

# ============================================
# PREDICTION FUNCTION
# ============================================
def make_prediction(model, scaler, input_df):
    """
    Make prediction using the model
    """
    try:
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"❌ Error membuat prediksi: {e}")
        return None, None

# ============================================
# GLUCOSE ANALYSIS FUNCTION
# ============================================
def analyze_glucose_level(glucose_value):
    """
    Analyze glucose level and return category
    """
    if glucose_value < 100:
        category = "Normal (Puasa)"
        color = "green"
        risk = "Rendah"
    elif glucose_value < 126:
        category = "Prediabetes (Puasa)"
        color = "orange"
        risk = "Sedang"
    elif glucose_value < 140:
        category = "Diabetes (Puasa)"
        color = "red"
        risk = "Tinggi"
    elif glucose_value < 200:
        category = "Prediabetes (2 jam)"
        color = "orange"
        risk = "Sedang"
    else:
        category = "Diabetes (2 jam)"
        color = "red"
        risk = "Tinggi"
    
    return category, color, risk

# ============================================
# MAIN APP
# ============================================

# TITLE
st.markdown('<h1 class="main-title">🩺 Aplikasi Prediksi Diabetes</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
Aplikasi ini memprediksi risiko diabetes berdasarkan data medis pasien menggunakan model Machine Learning 
yang telah dilatih dengan dataset Pima Indians Diabetes. <b>Range glukosa telah diperbaiki hingga 400 mg/dL</b> 
untuk mencakup nilai diabetes yang realistis.
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUT SECTION
# ============================================
input_df = get_user_input()

# Prediction button
st.sidebar.markdown("")
predict_button = st.sidebar.button(
    "🚀 **Lakukan Prediksi**", 
    type="primary", 
    use_container_width=True,
    help="Klik untuk menganalisis data dan mendapatkan prediksi"
)

# Clear history button
if st.session_state.prediction_history:
    clear_button = st.sidebar.button(
        "🗑️ **Clear History**", 
        use_container_width=True,
        help="Hapus riwayat prediksi"
    )
    if clear_button:
        st.session_state.prediction_history = []
        st.rerun()

# ============================================
# MAIN CONTENT AREA - INPUT DISPLAY
# ============================================

# Display input data
st.header("📋 **Data Input Pasien**")
col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(
        input_df.style.applymap(
            lambda x: 'background-color: #e8f4f8' if isinstance(x, (int, float)) else ''
        ),
        use_container_width=True
    )

with col2:
    st.metric("**Jumlah Fitur**", len(input_df.columns))
    st.metric("**Status Data**", "✅ Lengkap" if not input_df.isnull().any().any() else "⚠️ Tidak Lengkap")
    
    # Analisis glukosa cepat
    glucose_value = input_df['Glucose'].iloc[0]
    category, color, risk = analyze_glucose_level(glucose_value)
    st.metric("**Kategori Glukosa**", category)

st.markdown("---")

# ============================================
# GLUCOSE VISUALIZATION
# ============================================
st.subheader("📊 **Skala Glukosa**")

# Create glucose scale visualization
fig_glucose, ax_glucose = plt.subplots(figsize=(12, 3))

# Define ranges and colors
ranges = [
    (0, 100, "Normal\n(Puasa)", "#28a745"),
    (100, 126, "Prediabetes\n(Puasa)", "#ffc107"),
    (126, 140, "Diabetes\n(Puasa)", "#dc3545"),
    (140, 200, "Prediabetes\n(2 jam)", "#ffc107"),
    (200, 400, "Diabetes\n(2 jam)", "#dc3545")
]

# Plot each range
for start, end, label, color in ranges:
    ax_glucose.barh(0, end-start, left=start, height=0.6, color=color, alpha=0.7, edgecolor='black')
    ax_glucose.text((start+end)/2, 0, label, 
                   ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=9)

# Plot user's glucose value
user_glucose = glucose_value
ax_glucose.plot(user_glucose, 0, 'ko', markersize=12, label='Nilai Anda')
ax_glucose.plot(user_glucose, 0, 'wo', markersize=6)
ax_glucose.text(user_glucose, 0.8, f"{user_glucose} mg/dL", 
               ha='center', va='bottom', 
               fontweight='bold', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

# Configure plot
ax_glucose.set_xlim(0, 400)
ax_glucose.set_ylim(-1, 1.5)
ax_glucose.set_xlabel('Glukosa Plasma (mg/dL)', fontweight='bold')
ax_glucose.set_title('Skala Glukosa dan Kategori', fontweight='bold', fontsize=14, pad=20)
ax_glucose.get_yaxis().set_visible(False)
ax_glucose.legend(loc='upper right')
ax_glucose.grid(True, alpha=0.3, linestyle='--')

# Add reference lines
for threshold in [100, 126, 140, 200]:
    ax_glucose.axvline(x=threshold, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_glucose.text(threshold, -0.7, str(threshold), 
                   ha='center', va='top', 
                   fontsize=8, color='black', fontweight='bold')

st.pyplot(fig_glucose)

# Glucose interpretation
st.info(f"""
**Interpretasi Glukosa Anda ({user_glucose} mg/dL):**

**Jika nilai puasa (8 jam):**
- {'✅' if user_glucose < 100 else '⚠️'} **Normal**: < 100 mg/dL
- {'✅' if 100 <= user_glucose < 126 else '⚠️'} **Prediabetes**: 100-125 mg/dL  
- {'✅' if user_glucose >= 126 else '⚠️'} **Diabetes**: ≥ 126 mg/dL

**Jika nilai 2 jam setelah makan:**
- {'✅' if user_glucose < 140 else '⚠️'} **Normal**: < 140 mg/dL
- {'✅' if 140 <= user_glucose < 200 else '⚠️'} **Prediabetes**: 140-199 mg/dL
- {'✅' if user_glucose >= 200 else '⚠️'} **Diabetes**: ≥ 200 mg/dL
""")

st.markdown("---")

# ============================================
# PREDICTION SECTION
# ============================================
if predict_button:
    with st.spinner('🔍 Menganalisis data dan membuat prediksi...'):
        # Load model
        model, scaler = load_models()
        
        if model is not None and scaler is not None:
            # Make prediction
            prediction, probabilities = make_prediction(model, scaler, input_df)
            
            if prediction is not None:
                # Store in session state
                st.session_state.current_prediction = {
                    'data': input_df.copy(),
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'glucose': user_glucose,
                    'glucose_category': analyze_glucose_level(user_glucose)[0]
                }
                
                # Add to history
                st.session_state.prediction_history.append(
                    st.session_state.current_prediction.copy()
                )
                
                # Display results
                st.header("🎯 **Hasil Prediksi**")
                
                # Results in columns
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("📊 **Status Prediksi**")
                    if prediction == 0:
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ NON-DIABETES</h3>
                            <p>Berdasarkan analisis data, risiko diabetes Anda termasuk rendah.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown("""
                        <div class="error-box">
                            <h3>⚠️ DIABETES</h3>
                            <p>Berdasarkan analisis data, terdapat indikasi risiko diabetes.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.warning("**Disarankan untuk konsultasi dengan dokter!**")
                
                with result_col2:
                    st.subheader("📈 **Probabilitas**")
                    
                    # Create probability chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    classes = ['Non-Diabetes', 'Diabetes']
                    colors = ['#28a745', '#dc3545']
                    
                    bars = ax.bar(classes, probabilities, color=colors, alpha=0.8, edgecolor='black')
                    ax.set_ylabel('Probabilitas', fontsize=12, fontweight='bold')
                    ax.set_ylim(0, 1)
                    ax.set_title('Distribusi Probabilitas Prediksi', fontsize=14, fontweight='bold', pad=20)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2., 
                            height + 0.02,
                            f'{prob:.1%}',
                            ha='center', 
                            va='bottom',
                            fontsize=11,
                            fontweight='bold'
                        )
                    
                    # Add grid for better readability
                    ax.yaxis.grid(True, alpha=0.3)
                    ax.set_axisbelow(True)
                    
                    # Customize spines
                    for spine in ['top', 'right']:
                        ax.spines[spine].set_visible(False)
                    
                    st.pyplot(fig)
                    
                    # Display probability values
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric("Non-Diabetes", f"{probabilities[0]:.1%}")
                    with prob_col2:
                        st.metric("Diabetes", f"{probabilities[1]:.1%}")
                
                # ============================================
                # DETAILED GLUCOSE ANALYSIS
                # ============================================
                st.markdown("---")
                st.header("📝 **Analisis Detail Glukosa**")
                
                glucose_col1, glucose_col2, glucose_col3 = st.columns(3)
                
                with glucose_col1:
                    st.metric("Nilai Glukosa", f"{user_glucose} mg/dL")
                
                with glucose_col2:
                    category, color, risk = analyze_glucose_level(user_glucose)
                    st.metric("Kategori", category)
                
                with glucose_col3:
                    if user_glucose >= 200:
                        st.error("**Nilai Diabetes**")
                    elif user_glucose >= 140:
                        st.warning("**Nilai Prediabetes**")
                    else:
                        st.success("**Nilai Normal**")
                
                # Critical warning for high glucose
                if user_glucose >= 200:
                    st.error("""
                    🚨 **PERHATIAN: NILAI GLUKOSA TINGGI!**
                    
                    **Glukosa ≥ 200 mg/dL** termasuk kategori **DIABETES** berdasarkan standar medis.
                    
                    **Tindakan yang disarankan:**
                    1. **Segera konsultasi dengan dokter** spesialis penyakit dalam atau endokrinologi
                    2. Lakukan **pemeriksaan HbA1c** untuk konfirmasi diagnosis
                    3. Mulai **monitoring gula darah** rutin
                    4. Pertimbangkan **perubahan gaya hidup** segera
                    """)
                
                # ============================================
                # RECOMMENDATIONS BASED ON GLUCOSE LEVEL
                # ============================================
                st.markdown("---")
                st.header("💡 **Rekomendasi Berdasarkan Level Glukosa**")
                
                if user_glucose >= 200:  # DIABETES
                    st.markdown("""
                    ### 🏥 **Rekomendasi untuk Nilai Diabetes (≥ 200 mg/dL):**
                    
                    #### 1. **Konsultasi Medis Segera**
                    - 🔴 **PRIORITAS**: Buat janji dengan dokter dalam 1-2 minggu
                    - Lakukan pemeriksaan **HbA1c** (target: < 7%)
                    - Diskusikan kemungkinan perlu **obat oral atau insulin**
                    - Lakukan **pemeriksaan komplikasi** (mata, ginjal, saraf)
                    
                    #### 2. **Manajemen Darurat**
                    - Monitor gula darah **3-4 kali sehari**
                    - Waspada gejala **hiperglikemia** (haus berlebihan, sering buang air kecil, lemas)
                    - Siapkan **rencana darurat** jika gula darah > 300 mg/dL
                    
                    #### 3. **Perubahan Diet**
                    - Konsultasi dengan **ahli gizi**
                    - Hitung **kebutuhan kalori** harian
                    - Batasi **karbohidrat** < 45% total kalori
                    - Hindari **gula tambahan** dan makanan olahan
                    
                    #### 4. **Aktivitas Fisik**
                    - Olahraga **150 menit/minggu** (intensitas sedang)
                    - Latihan **kekuatan 2x/minggu**
                    - Hindari duduk terlalu lama
                    """)
                
                elif user_glucose >= 140:  # PREDIABETES
                    st.markdown("""
                    ### 🟡 **Rekomendasi untuk Nilai Prediabetes (140-199 mg/dL):**
                    
                    #### 1. **Intervensi Dini**
                    - Konsultasi dokter untuk **pencegahan progresi**
                    - Lakukan **pemeriksaan lanjutan** dalam 3-6 bulan
                    - Pertimbangkan **program pencegahan diabetes**
                    
                    #### 2. **Modifikasi Gaya Hidup**
                    - Turunkan **5-7% berat badan** jika overweight
                    - Tingkatkan aktivitas fisik **≥ 150 menit/minggu**
                    - Pilih **karbohidrat kompleks** (gandum utuh, sayuran)
                    
                    #### 3. **Monitoring**
                    - Cek gula darah **1-2 kali/minggu**
                    - Monitor **berat badan** mingguan
                    - Catat **asupan makanan** harian
                    
                    #### 4. **Edukasi**
                    - Ikuti **program edukasi diabetes**
                    - Pelajari **gejala diabetes**
                    - Pahami **faktor risiko**
                    """)
                
                else:  # NORMAL
                    st.markdown("""
                    ### 🟢 **Rekomendasi untuk Nilai Normal (< 140 mg/dL):**
                    
                    #### 1. **Pencegahan**
                    - Pertahankan **berat badan ideal**
                    - Lakukan **medical check-up tahunan**
                    - Monitor **faktor risiko** keluarga
                    
                    #### 2. **Gaya Hidup Sehat**
                    - Konsumsi **makanan seimbang**
                    - Tetap **aktif secara fisik**
                    - Kelola **stres** dengan baik
                    
                    #### 3. **Awareness**
                    - Kenali **gejala diabetes dini**
                    - Waspada jika ada **perubahan kesehatan**
                    - Edukasi keluarga tentang **pencegahan diabetes**
                    """)
                
                # ============================================
                # FEATURE IMPORTANCE
                # ============================================
                try:
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("---")
                        st.header("📊 **Kontribusi Fitur dalam Prediksi**")
                        
                        feature_names = ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit', 
                                       'Insulin', 'BMI', 'Riwayat Diabetes', 'Usia']
                        
                        feature_importance = pd.DataFrame({
                            'Fitur': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        colors_imp = ['#3498db' if 'Glukosa' in x else '#95a5a6' for x in feature_importance['Fitur']]
                        bars2 = ax2.barh(feature_importance['Fitur'], feature_importance['Importance'], 
                                        color=colors_imp, edgecolor='black', alpha=0.8)
                        ax2.set_xlabel('Tingkat Kepentingan', fontweight='bold', fontsize=12)
                        ax2.set_title('Kontribusi Fitur dalam Prediksi Diabetes', 
                                    fontweight='bold', fontsize=14, pad=20)
                        
                        # Add value labels
                        for bar in bars2:
                            width = bar.get_width()
                            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                                    f'{width:.3f}', 
                                    ha='left', va='center', fontweight='bold')
                        
                        # Highlight glucose
                        if 'Glukosa' in feature_importance['Fitur'].values:
                            glucose_idx = feature_importance[feature_importance['Fitur'] == 'Glukosa'].index[0]
                            bars2[glucose_idx].set_edgecolor('red')
                            bars2[glucose_idx].set_linewidth(2)
                        
                        ax2.grid(True, alpha=0.3, axis='x')
                        st.pyplot(fig2)
                        
                        st.info("""
                        **Interpretasi Feature Importance:**
                        - **Glukosa** (disorot biru) biasanya menjadi faktor paling penting
                        - Nilai lebih tinggi = kontribusi lebih besar dalam prediksi
                        - Model mempertimbangkan semua fitur secara holistik
                        """)
                except:
                    pass

# ============================================
# PREDICTION HISTORY
# ============================================
if st.session_state.prediction_history:
    st.markdown("---")
    st.header("📜 **Riwayat Prediksi**")
    
    for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:]), 1):
        with st.expander(f"Prediksi {i} - {pred['timestamp']} | Glukosa: {pred['glucose']} mg/dL"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write("**Data Input:**")
                st.dataframe(pred['data'].style.format({
                    'Glucose': '{:.0f} mg/dL',
                    'BMI': '{:.1f} kg/m²',
                    'BloodPressure': '{:.0f} mm Hg'
                }), use_container_width=True)
            
            with col2:
                st.write("**Hasil Prediksi:**")
                status = "Non-Diabetes ✅" if pred['prediction'] == 0 else "Diabetes ⚠️"
                st.metric("Status", status)
                st.metric("Glukosa", f"{pred['glucose']} mg/dL")
                st.metric("Kategori", pred['glucose_category'])
            
            with col3:
                st.write("**Probabilitas:**")
                st.metric("Non-Diabetes", f"{pred['probabilities'][0]:.1%}")
                st.metric("Diabetes", f"{pred['probabilities'][1]:.1%}")

# ============================================
# SIDEBAR - INFORMATION SECTION
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ **Informasi Model & Glukosa**")

with st.sidebar.expander("**📊 Kriteria Diagnosis Diabetes**"):
    st.markdown("""
    ### **Berdasarkan American Diabetes Association (ADA):**
    
    **1. Diabetes:**
    - Glukosa puasa ≥ 126 mg/dL
    - Glukosa 2 jam ≥ 200 mg/dL (OGTT)
    - HbA1c ≥ 6.5%
    - Gejala klasik + glukosa acak ≥ 200 mg/dL
    
    **2. Prediabetes:**
    - Glukosa puasa: 100-125 mg/dL
    - Glukosa 2 jam: 140-199 mg/dL
    - HbA1c: 5.7-6.4%
    
    **3. Normal:**
    - Glukosa puasa < 100 mg/dL
    - Glukosa 2 jam < 140 mg/dL
    - HbA1c < 5.7%
    """)

with st.sidebar.expander("**🤖 Detail Teknis Model**"):
    st.markdown("""
    **Model Machine Learning:**
    - Algoritma: Random Forest Classifier
    - Akurasi: ~77%
    - Dataset: Pima Indians Diabetes
    - Jumlah Fitur: 8 parameter medis
    - Total Samples: 768
    - Training-Testing Split: 80%-20%
    
    **Fitur yang Digunakan:**
    1. Kehamilan (0-20)
    2. Glukosa (50-400 mg/dL) ✅ **Diperbarui**
    3. Tekanan Darah (40-180 mm Hg)
    4. Ketebalan Kulit (0-99 mm)
    5. Insulin (0-1000 µU/mL)
    6. BMI (10-60 kg/m²)
    7. Riwayat Diabetes Keluarga (0.08-2.50)
    8. Usia (0-100 tahun)
    """)

with st.sidebar.expander("**⚠️ Batasan Model**"):
    st.markdown("""
    - Model dilatih pada populasi spesifik (Pima Indians)
    - Tidak menggantikan diagnosis dokter profesional
    - Akurasi terbatas (~77%)
    - Tidak mempertimbangkan semua faktor klinis
    - Tidak termasuk data laboratorium lengkap
    - Konsultasi dokter tetap diperlukan untuk diagnosis pasti
    """)

# ============================================
# DISCLAIMER
# ============================================
st.sidebar.markdown("---")
st.sidebar.warning("""
⚠️ **DISCLAIMER MEDIS:**

**APLIKASI INI UNTUK TUJUAN EDUKASI DAN SKRINING AWAL SAJA.**

**TIDAK MENGGANTIKAN DIAGNOSIS DOKTER PROFESIONAL.**

**KONSULTASI DOKTER DIPERLUKAN UNTUK:**
- Diagnosis yang akurat
- Pemeriksaan laboratorium lengkap
- Rencana pengobatan yang tepat
- Monitoring berkala

**Nilai glukosa ≥ 200 mg/dL memerlukan evaluasi medis segera.**
""")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: #666; font-size: 0.9rem;'>
    🩺 <b>Aplikasi Prediksi Diabetes</b> | 
    <b>Range Glukosa Diperbarui: 50-400 mg/dL</b> | 
    Dibuat dengan ❤️ menggunakan Streamlit
    </p>
    <p style='color: #888; font-size: 0.8rem; margin-top: -10px;'>
    © 2024 | Versi 2.0 | Untuk Edukasi Kesehatan | Glukosa ≥ 200 mg/dL = Kategori Diabetes
    </p>
</div>
""", unsafe_allow_html=True)