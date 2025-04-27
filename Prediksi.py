import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Memuat model dan komponen preprocessing
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl") 

@st.cache_data
def load_data():
    df = pd.read_csv("data_fix.csv", encoding='utf-8')
    df.columns = df.columns.str.strip() 
    return df

df = load_data()

st.title("Student Dropout Prediction App")
st.sidebar.header("User Input Features")

input_data = {}

# Menentukan fitur kategorikal dan numerikal
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

# Input untuk fitur kategorikal
for col in categorical_columns:
    options = df[col].unique().tolist()
    input_data[col] = st.sidebar.selectbox(f"{col}", options)

# Input untuk fitur numerikal
for col in numerical_columns:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    input_data[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=min_val)

# Mengonversi data input menjadi DataFrame
input_df = pd.DataFrame([input_data])

# Mengkodekan fitur kategorikal dengan encoder yang sudah disimpan
for col in categorical_columns:
    if col in encoder:
        try:
            input_df[col] = encoder[col].transform(input_df[col])
        except ValueError:
            st.error(f"Nilai yang dimasukkan untuk {col} tidak ada dalam encoder. Gunakan nilai yang tersedia.")
            st.stop()

# Gabungkan kembali fitur kategorikal dan numerik
processed_columns = numerical_columns + categorical_columns  # Pastikan urutan sesuai data latih
input_df = input_df.reindex(columns=processed_columns, fill_value=0)

# Melakukan scaling pada fitur numerikal
try:
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
except ValueError as e:
    st.error(f"Kesalahan dalam scaling: {e}")
    st.stop()

# Tombol prediksi
if st.sidebar.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]  # Melakukan prediksi
        prediction_proba = model.predict_proba(input_df)[0]  # Mendapatkan probabilitas prediksi
        st.write(f"## Prediction: {'Dropout' if prediction == 1 else 'Continue'}")
        st.write(f"### Confidence: {max(prediction_proba) * 100:.2f}%")
    except Exception as e:
        st.error(f"Kesalahan saat melakukan prediksi: {e}")

# Menampilkan gambaran umum dataset
st.subheader("Dataset Overview")
st.write(df.head())

# Memvisualisasikan distribusi variabel target
if 'Curricular_units_1st_sem_enrolled' in df.columns:
    fig = px.histogram(df, x=df['Curricular_units_1st_sem_enrolled'], title='Distribusi Status Mahasiswa')
    st.plotly_chart(fig)
else:
    st.warning("⚠️ Kolom 'Curricular_units_1st_sem_enrolled' tidak ditemukan! Periksa kembali dataset yang digunakan.")
    st.write("Kolom yang tersedia dalam dataset:", df.columns.tolist())
