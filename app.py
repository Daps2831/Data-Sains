import streamlit as st
import pandas as pd
import joblib

#================================================================================
# BAGIAN 1: FUNGSI DAN PEMETAAN (SESUAI ATURAN BARU ANDA)
#================================================================================

def preprocess_new_data(input_data):
    """
    Menerima input data dalam bentuk dictionary,
    melakukan encoding dan scaling sesuai aturan baru,
    dan mengembalikan data yang siap untuk prediksi.
    """
    try:
        loaded_preprocessing = joblib.load('preprocessing_objects.joblib')
        scaler = loaded_preprocessing['scaler']
    except FileNotFoundError:
        st.error("File 'preprocessing_objects.joblib' tidak ditemukan.")
        return None

    df = pd.DataFrame([input_data])

    # 1. Definisikan pemetaan sesuai klarifikasi Anda
    gender_map = {'Female': 0, 'Male': 1}
    # Untuk kolom biner, kita akan buat pemetaan terpisah untuk UI
    yes_no_map = {'No': 0, 'Yes': 1}
    # Menggunakan 'No' sebagai representasi user-friendly untuk '0'
    caec_map = {'No': 3, 'Frequently': 1, 'Sometimes': 2, 'Always': 0}
    calc_map = {'No': 2, 'Frequently': 1, 'Sometimes': 0}
    mtrans_map = {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4}

    # 2. Definisikan daftar kolom sesuai klarifikasi Anda
    numerical_cols = [
        'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC',
        'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE'
    ]
    # 'CALC' & 'MTRANS' tidak ada di daftar numerik Anda, jadi tidak di-scale
    feature_order = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
        'CALC', 'MTRANS'
    ]

    # 3. Encoding
    # Kolom kategorikal utama
    df['Gender'] = df['Gender'].map(gender_map)
    df['CAEC'] = df['CAEC'].map(caec_map)
    df['CALC'] = df['CALC'].map(calc_map)
    df['MTRANS'] = df['MTRANS'].map(mtrans_map)

    # Ubah kolom biner (Yes/No) menjadi numerik (1/0)
    # sebelum di-scaling bersama kolom numerik lainnya
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map(yes_no_map)
    df['FAVC'] = df['FAVC'].map(yes_no_map)
    df['SMOKE'] = df['SMOKE'].map(yes_no_map)
    df['SCC'] = df['SCC'].map(yes_no_map)

    # 4. Scaling
    # Terapkan scaler HANYA pada daftar numerical_cols yang sudah Anda definisikan
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # 5. Atur Ulang Urutan Kolom
    df = df[feature_order]

    return df

# Pemetaan untuk menerjemahkan output prediksi (sudah benar dari sebelumnya)
label_mapping_target = {
    0: 'Normal Weight', 1: 'Insufficient Weight', 2: 'Obesity Type I',
    3: 'Obesity Type II', 4: 'Obesity Type III', 5: 'Overweight Level I',
    6: 'Overweight Level II'
}

#================================================================================
# BAGIAN 2: ANTARMUKA APLIKASI STREAMLIT (UI disesuaikan)
#================================================================================

try:
    model = joblib.load('random_forest_obesity_model.joblib')
except FileNotFoundError:
    st.error("File 'random_forest_obesity_model.joblib' tidak ditemukan.")
    model = None

st.title('Prediksi Tingkat Obesitas')
st.write('Aplikasi ini memprediksi tingkat obesitas berdasarkan kebiasaan gaya hidup dan atribut fisik.')

col1, col2 = st.columns(2)

with col1:
    st.header("Atribut Fisik & Riwayat")
    age = st.number_input('Umur (Tahun)', min_value=1, max_value=100, value=25)
    gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    height = st.number_input('Tinggi Badan (meter)', min_value=1.0, max_value=2.5, value=1.75, format="%.2f")
    weight = st.number_input('Berat Badan (kg)', min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
    family_history = st.selectbox('Riwayat obesitas dalam keluarga?', ['Yes', 'No'])

with col2:
    st.header("Kebiasaan Konsumsi")
    favc = st.selectbox('Sering konsumsi makanan tinggi kalori (FAVC)?', ['Yes', 'No'])
    fcvc = st.slider('Frekuensi makan sayur (FCVC)', 1.0, 3.0, 2.0, step=1.0)
    ncp = st.slider('Jumlah makan utama per hari (NCP)', 1.0, 4.0, 3.0, step=1.0)
    # Opsi disesuaikan dengan map Anda
    caec = st.selectbox('Konsumsi makanan di antara waktu makan (CAEC)?', ['No', 'Sometimes', 'Frequently', 'Always'])
    calc = st.selectbox('Konsumsi alkohol (CALC)?', ['No', 'Sometimes', 'Frequently'])
    ch2o = st.slider('Konsumsi air per hari (liter) (CH2O)', 1.0, 3.0, 2.0, step=1.0)

st.header("Aktivitas & Gaya Hidup")
smoke = st.selectbox('Apakah Anda merokok (SMOKE)?', ['Yes', 'No'])
scc = st.selectbox('Konsumsi minuman berkalori?', ['Yes', 'No'])
faf = st.slider('Frekuensi aktivitas fisik per minggu (FAF)', 0.0, 3.0, 1.0, step=1.0)
tue = st.slider('Waktu menggunakan gawai per hari (TUE)', 0.0, 2.0, 1.0, step=1.0)
mtrans = st.selectbox('Moda transportasi utama (MTRANS)', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])

if st.button('Prediksi Tingkat Obesitas'):
    if model is not None:
        input_data = {
            'Age': age, 'Gender': gender, 'Height': height, 'Weight': weight,
            'family_history_with_overweight': family_history, 'FAVC': favc,
            'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke,
            'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue,
            'CALC': calc, 'MTRANS': mtrans
        }
        processed_data = preprocess_new_data(input_data)
        if processed_data is not None:
            prediction_numeric = model.predict(processed_data)[0]
            prediction_text = label_mapping_target[prediction_numeric]
            st.success(f"Hasil Prediksi: **{prediction_text}**")