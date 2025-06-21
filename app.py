import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="wide")
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa Pada Suatu Mata Kuliah dengan Decision Tree & SVM")

# Upload file CSV
uploaded_file = st.file_uploader("📂 Upload Dataset Mahasiswa (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    expected_cols = {"Nama", "NIM", "Kehadiran", "UTS", "UAS", "Tugas"}
    if not expected_cols.issubset(df.columns):
        st.error("❌ Kolom tidak sesuai. Pastikan: Nama, NIM, Kehadiran, UTS, UAS, Tugas")
        st.stop()

    # Preview Dataset
    st.subheader("🗂️ Preview Dataset")
    st.dataframe(df.head())

    st.markdown("### ℹ️ Ringkasan Dataset")
    st.write(f"- Jumlah Mahasiswa: **{df.shape[0]}**")
    st.write(f"- Jumlah Kolom: **{df.shape[1]}**")
    st.markdown("#### Statistik Deskriptif:")
    st.dataframe(df.describe())

    # Buat label kelulusan otomatis
    df["Lulus"] = np.where(
        (df["Kehadiran"] >= 75) &
        (df["Tugas"] >= 70) &
        (df["UTS"] >= 70) &
        (df["UAS"] >= 75), 1, 0
    )

    # Fitur dan Label
    fitur = ["Kehadiran", "UTS", "UAS", "Tugas"]
    X = df[fitur]
    y = df["Lulus"]

    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Models
    model_dt = DecisionTreeClassifier(random_state=42)
    model_dt.fit(X_train, y_train)

    model_svm = SVC(probability=True, random_state=42)
    model_svm.fit(X_train, y_train)

    st.success("✅ Model Decision Tree dan SVM berhasil dilatih dengan dataset.")

    # Input prediksi
    st.subheader("🧪 Prediksi Mahasiswa Baru")
    with st.form("form_predict"):
        kehadiran = st.slider("Kehadiran (%)", 0, 100, 75)
        uts = st.slider("Nilai UTS", 0, 100, 70)
        uas = st.slider("Nilai UAS", 0, 100, 75)
        tugas = st.slider("Nilai Tugas", 0, 100, 75)
        threshold = st.slider("Threshold Probabilitas Kelulusan (%)", 0, 100, 50)
        submit = st.form_submit_button("🔮 Prediksi")

    if submit:
        input_data = np.array([[kehadiran, uts, uas, tugas]])
        input_scaled = scaler.transform(input_data)

        proba_dt = model_dt.predict_proba(input_scaled)[0][1]
        proba_svm = model_svm.predict_proba(input_scaled)[0][1]

        hasil_dt = "Lulus" if proba_dt >= threshold / 100 else "Tidak Lulus"
        hasil_svm = "Lulus" if proba_svm >= threshold / 100 else "Tidak Lulus"

        st.subheader("📈 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Decision Tree", hasil_dt)
            st.caption(f"Probabilitas: {proba_dt:.2f}")
        with col2:
            st.metric("SVM", hasil_svm)
            st.caption(f"Probabilitas: {proba_svm:.2f}")

        # Penjelasan dan Saran
        st.subheader("🧠 Penjelasan & Rekomendasi")
        if hasil_dt == hasil_svm:
            st.info(f"📌 Kedua algoritma sepakat bahwa mahasiswa ini kemungkinan **{hasil_dt}**.")
        else:
            st.warning("⚠️ Kedua algoritma memberikan prediksi yang **berbeda**.")

        if hasil_dt == "Tidak Lulus" or hasil_svm == "Tidak Lulus":
            st.error("💡 Saran: Tingkatkan kehadiran dan nilai tugas/UTS/UAS. Perbanyak latihan dan diskusi materi.")
        else:
            st.success("👍 Tetap pertahankan prestasi. Terus belajar dan bantu teman yang kesulitan!")

    # Evaluasi
    st.subheader("📊 Evaluasi Model")
    y_pred_dt = model_dt.predict(X_test)
    y_pred_svm = model_svm.predict(X_test)

    acc_dt = accuracy_score(y_test, y_pred_dt)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Decision Tree")
        st.write(f"Akurasi: **{acc_dt:.2f}**")
        fig1, ax1 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="YlGn", ax=ax1)
        ax1.set_title("Confusion Matrix - Decision Tree")
        st.pyplot(fig1)

    with col2:
        st.markdown("#### SVM")
        st.write(f"Akurasi: **{acc_svm:.2f}**")
        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="PuBu", ax=ax2)
        ax2.set_title("Confusion Matrix - SVM")
        st.pyplot(fig2)

    # Penjelasan Akurasi
    st.subheader("📌 Perbandingan Akurasi dan Penjelasan")
    st.write(f"🎯 **Decision Tree Akurasi:** `{acc_dt:.2f}`")
    st.write(f"🎯 **SVM Akurasi:** `{acc_svm:.2f}`")

    if acc_dt > acc_svm:
        st.success("✅ Model Decision Tree lebih akurat dalam dataset ini. Cocok digunakan untuk interpretasi keputusan.")
    elif acc_dt < acc_svm:
        st.success("✅ Model SVM lebih akurat. Cocok untuk pola yang lebih kompleks dan margin pemisah yang optimal.")
    else:
        st.info("⚖️ Kedua model memiliki akurasi yang sama. Bisa digunakan bergantian sesuai konteks.")

else:
    st.info("Silakan upload file CSV dengan kolom: Nama, NIM, Kehadiran, UTS, UAS, Tugas.")
