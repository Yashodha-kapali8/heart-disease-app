import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")   # trained on 5 numeric columns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("### 🧠 AI-powered health risk prediction")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 25)
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_ecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------- ENCODING ----------------

fasting_bs_val = 1 if fasting_bs == "Yes" else 0

# Chest Pain One-Hot
cp_asy = 1 if chest_pain == "ASY" else 0
cp_ata = 1 if chest_pain == "ATA" else 0
cp_nap = 1 if chest_pain == "NAP" else 0
cp_ta  = 1 if chest_pain == "TA" else 0

# ECG One-Hot
ecg_lvh = 1 if resting_ecg == "LVH" else 0
ecg_normal = 1 if resting_ecg == "Normal" else 0
ecg_st = 1 if resting_ecg == "ST" else 0

# ST Slope Encoding (IMPORTANT)
st_slope_val = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Risk"):

    # Scale ONLY numeric features
    numeric_data = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak]])
    scaled = scaler.transform(numeric_data)

    # Combine all features (must match training order)
    final_data = np.array([[ 
        scaled[0][0],   # Age
        scaled[0][1],   # BP
        scaled[0][2],   # Cholesterol
        fasting_bs_val,
        scaled[0][3],   # MaxHR
        scaled[0][4],   # Oldpeak
        st_slope_val,
        cp_asy, cp_ata, cp_nap, cp_ta,
        ecg_lvh, ecg_normal, ecg_st
    ]])

    prediction = model.predict(final_data)

    # ---------------- OUTPUT ----------------
    st.markdown("---")

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
        st.markdown("**Advice:** Please consult a doctor immediately.")
    else:
        st.success("Low Risk of Heart Disease")
        st.markdown("**Advice:** Maintain a healthy lifestyle.")
