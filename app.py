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
        /* Background gradient (wine inspired) */
        .stApp {
            background: linear-gradient(160deg, #2c0f0f, #0d0d0d 90%);
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Center everything */
        .block-container {
            max-width: 1200px;
            margin: auto;
            padding-top: 3vh;
            padding-bottom: 5vh;
        }

        /* Title */
        h1 {
            font-size: 60px !important;
            text-align: center;
            font-weight: 900;
            margin-bottom: 0.3em;
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
            background: rgba(255, 255, 255, 0.07);
            border-radius: 18px;
            padding: 25px;
            box-shadow: 0px 8px 30px rgba(0,0,0,0.9);
            backdrop-filter: blur(12px);
            margin-bottom: 20px;
        }

        /* Sliders */
        div[data-baseweb="slider"] > div {
            height: 16px !important;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
        }
        div[data-baseweb="slider"] span {
            height: 28px !important;
            width: 28px !important;
            background: #ff4b4b !important;
            border: 3px solid white !important;
            border-radius: 50%;
            box-shadow: 0px 0px 15px #ff4b4b;
        }
        label {
            font-size: 20px !important;
            font-weight: 600 !important;
            color: #f5f5f5 !important;
        }

        /* Button */
        .stButton > button {
            background: linear-gradient(135deg, #ff4b4b, #b22222);
            color: white !important;
            font-size: 26px !important;
            font-weight: 700 !important;
            padding: 1em 2em;
            border-radius: 14px;
            border: none;
            width: 100%;
            transition: all 0.3s ease-in-out;
            box-shadow: 0px 5px 20px rgba(255,75,75,0.6);
        }
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0px 8px 30px rgba(255,75,75,0.9);
        }

        /* Result cards */
        .result-card {
            padding: 2em;
            margin: 2em auto;
            border-radius: 20px;
            text-align: center;
            font-size: 2em !important;
            font-weight: 800 !important;
            width: 85%;
            animation: fadeIn 1s ease-in-out;
        }
        .good {
            background: rgba(0, 128, 0, 0.2);
            color: #98fb98 !important;
            border: 3px solid #32cd32;
            box-shadow: 0px 0px 25px rgba(50,205,50,0.7);
        }
        .bad {
            background: rgba(178,34,34,0.2);
            color: #ff9999 !important;
            border: 3px solid #ff4b4b;
            box-shadow: 0px 0px 25px rgba(255,75,75,0.7);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h1>Wine Quality Predictor 🍷</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Adjust the chemistry sliders below and predict your wine's quality.</p>", unsafe_allow_html=True)

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
