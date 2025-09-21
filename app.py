import streamlit as st
import joblib
import numpy as np

# ================== Page Config ==================
st.set_page_config(
    page_title="Wine Quality Predictor 🍷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Load Model & Scaler ==================
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        /* Full dark gradient background */
        .stApp {
            background: linear-gradient(135deg, #0d0d0d, #1c1c1c, #2a2a2a);
            color: #ffffff;
        }

        /* Center main block */
        .main-container {
            max-width: 700px;
            margin: auto;
            padding: 40px;
            background: rgba(20, 20, 20, 0.9);
            border-radius: 18px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.8);
        }

        h1 {
            color: #ff4d4d !important;
            font-size: 52px !important;
            text-align: center;
            margin-bottom: 10px;
        }

        p {
            text-align: center;
            font-size: 20px !important;
            color: #cccccc !important;
            margin-bottom: 30px;
        }

        /* Sliders */
        .stSlider label {
            font-size: 18px !important;
            color: #ffffff !important;
        }
        .stSlider [role='slider'] {
            height: 24px;
            border-radius: 12px;
            background: #ff4d4d;
        }
        .stSlider > div > div {
            background: #444 !important;
            height: 10px;
            border-radius: 5px;
        }

        /* Prediction Button */
        .stButton>button {
            background-color: #ff4d4d;
            color: white;
            border-radius: 12px;
            height: 65px;
            width: 100%;
            font-size: 22px;
            font-weight: bold;
            border: none;
            box-shadow: 0px 4px 15px rgba(255,77,77,0.4);
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #e63939;
            transform: scale(1.02);
        }

        /* Prediction Results */
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

# ================== App Layout ==================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("## 🍷 Wine Quality Predictor")
st.write("<p>Adjust the sliders below and see if your wine passes the test</p>", unsafe_allow_html=True)

# ================== Input Sliders ==================
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.07)
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

st.markdown("</div>", unsafe_allow_html=True)
