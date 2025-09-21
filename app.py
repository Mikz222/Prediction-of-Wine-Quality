import streamlit as st
import joblib
import numpy as np
from PIL import Image

# ================== Page Config ==================
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Load Model & Scaler ==================
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        body {
            background-color: #121212;
        }
        .stApp {
            background-color: #121212;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #ffffff !important;
            font-size: 48px !important;
            text-align: center;
            margin-bottom: 20px;
        }
        p, label {
            font-size: 20px !important;
            color: #cccccc !important;
        }
        .stSlider > div > div {
            padding: 10px 0;
        }
        .stSlider [role='slider'] {
            height: 20px;
            border-radius: 10px;
            background: #8e44ad;
        }
        .css-1offfwp {  /* Slider number text */
            font-size: 20px !important;
        }
        .stButton>button {
            background-color: #8e44ad;
            color: white;
            border-radius: 12px;
            height: 60px;
            font-size: 24px;
            font-weight: bold;
        }
        .result-good {
            color: #2ecc71;
            font-size: 32px;
            text-align: center;
            font-weight: bold;
            margin-top: 30px;
        }
        .result-bad {
            color: #e74c3c;
            font-size: 32px;
            text-align: center;
            font-weight: bold;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("## 🍷 Wine Quality Prediction")
st.write("<p style='text-align:center; font-size:22px;'>Enter the wine attributes below to predict its quality.</p>", unsafe_allow_html=True)

# ================== Input Sliders ==================
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.07)

with col2:
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46)
    density = st.slider("Density", 0.990, 1.005, 0.996, step=0.001)
    pH = st.slider("pH", 2.5, 4.5, 3.3)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65)
    alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.0)

# ================== Prediction Button ==================
if st.button("🍷 Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.markdown(f"<div class='result-good'>✅ This wine is predicted to be Good Quality<br>Confidence: {probability[1]*100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-bad'>❌ This wine is predicted to be Not Good Quality<br>Confidence: {probability[0]*100:.2f}%</div>", unsafe_allow_html=True)
