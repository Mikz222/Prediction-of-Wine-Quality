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
            background: linear-gradient(160deg, #2c0f0f, #0d0d0d 90%);
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Remove default Streamlit padding */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 2rem !important;
            max-width: 1200px;
            margin: auto;
        }

        /* Title */
        h1 {
            font-size: 56px !important;
            text-align: center;
            font-weight: 900;
            margin-top: 1rem;
            margin-bottom: 0.2em;
            background: linear-gradient(90deg, #ff4b4b, #ffaaaa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Subheader */
        .subheader {
            text-align: center;
            font-size: 22px !important;
            color: #dddddd;
            margin-bottom: 40px;
        }

        /* Glassmorphism card */
        .glass-card {
            background: rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.6);
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }

        /* Clean sliders (flat style, no glow) */
        div[data-baseweb="slider"] > div {
            height: 8px !important;
            background: #444 !important;
            border-radius: 6px;
        }
        div[data-baseweb="slider"] span {
            height: 18px !important;
            width: 18px !important;
            background: #ff4b4b !important;
            border: none !important;
            border-radius: 50%;
            box-shadow: none !important;
        }
        label {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #f5f5f5 !important;
        }

        /* Button */
        .stButton > button {
            background: linear-gradient(135deg, #ff4b4b, #b22222);
            color: white !important;
            font-size: 22px !important;
            font-weight: 700 !important;
            padding: 0.9em 2em;
            border-radius: 12px;
            border: none;
            width: 100%;
            transition: all 0.25s ease-in-out;
            box-shadow: 0px 4px 12px rgba(255,75,75,0.4);
        }
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0px 6px 20px rgba(255,75,75,0.7);
        }

        /* Result cards */
        .result-card {
            padding: 1.8em;
            margin: 2em auto;
            border-radius: 16px;
            text-align: center;
            font-size: 1.6em !important;
            font-weight: 700 !important;
            width: 80%;
            animation: fadeIn 0.8s ease-in-out;
        }
        .good {
            background: rgba(0, 128, 0, 0.15);
            color: #90ee90 !important;
            border: 2px solid #32cd32;
        }
        .bad {
            background: rgba(178,34,34,0.15);
            color: #ff9999 !important;
            border: 2px solid #ff4b4b;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h1>Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Adjust the sliders below and see if your wine passes the test 🍷</p>", unsafe_allow_html=True)

# ================== Input Layout ==================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.07)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46)
    density = st.slider("Density", 0.990, 1.005, 0.996, step=0.001)
    pH = st.slider("pH", 2.5, 4.5, 3.3)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65)
    alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.0)
    st.markdown('</div>', unsafe_allow_html=True)

# ================== Prediction ==================
if st.button("🔮 Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.markdown(
            f'<div class="result-card good">✅ Premium Wine Detected!<br>This wine is <b>Good Quality</b> 🍷<br><br>Confidence: {probability[1]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-card bad">❌ Needs Refinement...<br>This wine is <b>Not Good Quality</b> 🍷<br><br>Confidence: {probability[0]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
