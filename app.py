import streamlit as st
import joblib
import numpy as np

# ================== Page Config ==================
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Load Model & Scaler ==================
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background-color: #0d0d0d;
            color: #ffffff;
        }
        
        /* Header */
        h1, h2, h3 {
            color: #ffffff !important;
            font-size: 52px !important;
            text-align: center;
            margin-bottom: 10px;
            font-weight: 700;
        }

        /* Subheading */
        .subheader {
            text-align: center;
            font-size: 22px !important;
            color: #bbbbbb !important;
            margin-bottom: 30px;
        }

        /* Sliders container as card */
        .stSlider {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }

        /* Slider customization */
        .stSlider [role='slider'] {
            height: 18px;
            border-radius: 10px;
            background: linear-gradient(90deg, #8e44ad, #ff4b2b);
        }
        .css-1offfwp {
            font-size: 20px !important;
        }

        /* Button */
        .stButton>button {
            background: linear-gradient(135deg, #8e44ad, #c0392b);
            color: white;
            border-radius: 12px;
            height: 65px;
            font-size: 24px;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, #9b59b6, #e74c3c);
        }

        /* Results */
        .result-good {
            color: #2ecc71;
            font-size: 34px;
            text-align: center;
            font-weight: bold;
            margin-top: 40px;
        }
        .result-bad {
            color: #e74c3c;
            font-size: 34px;
            text-align: center;
            font-weight: bold;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("## 🍷 Wine Quality Prediction")
st.markdown("<p class='subheader'>Adjust the sliders to enter wine attributes and predict its quality.</p>", unsafe_allow_html=True)

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
        st.markdown(
            f"<div class='result-good'>✅ This wine is predicted to be Good Quality<br>Confidence: {probability[1]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-bad'>❌ This wine is predicted to be Not Good Quality<br>Confidence: {probability[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
