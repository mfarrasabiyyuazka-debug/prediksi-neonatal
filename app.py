import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)

st.set_page_config(page_title="Neonatal Mortality Predictor", layout="wide")

st.title("🏥 Sistem Prediksi Risiko Kematian Neonatal (Optimized)")

@st.cache_data
def load_data():
    df = pd.read_csv('neonatal_mortality_dataset (2).csv')
    return df

try:
    df = load_data()

    # --- PREPROCESSING ---
    compl_options = df['Delivery_Complications'].unique().tolist()
    place_options = df['Place_of_Delivery'].unique().tolist()

    df_encoded = pd.get_dummies(df, columns=['Delivery_Complications', 'Place_of_Delivery'])
    le = LabelEncoder()
    df_encoded['Neonatal_Outcome'] = le.fit_transform(df_encoded['Neonatal_Outcome'])

    X = df_encoded.drop('Neonatal_Outcome', axis=1)
    y = df_encoded['Neonatal_Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- SIDEBAR KONFIGURASI (DENGAN PEMBATASAN) ---
    st.sidebar.header("⚙️ Konfigurasi Model")
    model_name = st.sidebar.selectbox("Pilih Algoritma", ["Random Forest", "Logistic Regression"])

    if model_name == "Random Forest":
        # Perbaikan: Menambahkan batasan max_depth untuk mencegah overfitting
        n_trees = st.sidebar.slider("Jumlah Pohon", 10, 200, 100)
        depth = st.sidebar.slider("Kedalaman Maksimal (max_depth)", 1, 10, 5) # Default dibatasi ke 5
        model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, random_state=42)
    else:
        model = LogisticRegression()

    # Training
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # --- INPUT PREDIKSI ---
    st.write("### 📝 Masukkan Data Medis Baru")
    c1, c2 = st.columns(2)
    with c1:
        bw = st.number_input("Berat Lahir (kg)", 0.5, 5.0, 2.5)
        ga = st.number_input("Usia Kehamilan (minggu)", 20, 45, 36)
        ma = st.number_input("Usia Ibu (tahun)", 15, 50, 28)
    with c2:
        av = st.number_input("Jumlah Kunjungan Antenatal", 0, 15, 4)
        dc = st.selectbox("Komplikasi Persalinan", compl_options)
        pd_loc = st.selectbox("Lokasi Persalinan", place_options)
        bi = st.number_input("Inisiasi Menyusui (jam)", 0, 48, 1)

    if st.button("🚀 Prediksi Sekarang"):
        input_raw = pd.DataFrame([[bw, ga, ma, av, bi]], columns=['Birth_Weight_kg', 'Gestational_Age_weeks', 'Maternal_Age_years', 'Antenatal_Visits', 'Breastfeeding_Initiation_hrs'])
        for opt in compl_options: input_raw[f'Delivery_Complications_{opt}'] = 1 if dc == opt else 0
        for opt in place_options: input_raw[f'Place_of_Delivery_{opt}'] = 1 if pd_loc == opt else 0
        
        input_raw = input_raw[X.columns]
        input_val_scaled = scaler.transform(input_raw)
        res = model.predict(input_val_scaled)
        prob = model.predict_proba(input_val_scaled)
        res_text = le.inverse_transform(res)[0]

        if res_text == 'Alive': st.success(f"### Hasil Prediksi: **{res_text}**")
        else: st.error(f"### Hasil Prediksi: **{res_text}** (Risiko Tinggi)")
        st.write(f"Confidence Level: {np.max(prob)*100:.2f}%")

    # --- EVALUASI YANG LEBIH VALID ---
    st.write("---")
    st.write(f"### 📊 Evaluasi Performa Realistis ({model_name})")

    # Menambahkan Cross-Validation Score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuracy (Test)", f"{accuracy_score(y_test, y_pred):.2%}")
    col_m2.metric("CV Mean Accuracy", f"{cv_scores.mean():.2%}")
    col_m3.metric("Recall (Sensitivity)", f"{recall_score(y_test, y_pred):.2%}")
    col_m4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")

    # Penjelasan CV
    st.info(f"**Info:** CV Mean Accuracy ({cv_scores.mean():.2%}) adalah nilai rata-rata akurasi pada 5 lipatan data yang berbeda. Jika nilai ini jauh di bawah Accuracy Test, model Anda mengalami overfitting.")

    # Confusion Matrix
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    st.pyplot(fig)

except FileNotFoundError:
    st.error("File dataset tidak ditemukan.")