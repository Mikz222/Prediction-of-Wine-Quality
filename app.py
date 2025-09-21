import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

# --- Custom CSS for Fullscreen Premium UI ---
st.markdown(
    """
    <style>
    /* Remove default padding */
    .block-container {
        padding: 0 !important;
        margin: 0 auto;
        max-width: 100% !important;
    }

    /* Fullscreen background */
    .main {
        background: linear-gradient(135deg, #2a0a0a 0%, #0d0d0d 100%) !important;
        color: #f5f5f5;
        min-height: 100vh;
    }

    /* Title */
    h1, h2, h3 {
        font-family: 'Georgia', serif;
        text-align: center;
        font-weight: 700;
    }
    h2 {
        color: #ffcccc;
        margin-bottom: 0.2em;
        padding-top: 1em;
    }
    p.subtitle {
        text-align: center;
        color: #ddd;
        font-size: 1.1em;
        margin-bottom: 2em;
    }

    /* Glassmorphic input card */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 3em;
        border-radius: 18px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        width: 80%;
        margin: 0 auto 2em auto;
    }

    /* Sliders */
    div[data-baseweb="slider"] > div {
        height: 12px !important;
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
    }
    div[data-baseweb="slider"] span {
        height: 26px !important;
        width: 26px !important;
        background: #ff4b4b !important;
        border: 2px solid white !important;
        border-radius: 50%;
        box-shadow: 0px 0px 12px #ff4b4b;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #b22222, #ff4b4b);
        color: white;
        font-size: 20px;
        font-weight: 600;
        padding: 0.8em 1.6em;
        border-radius: 12px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff4b4b, #b22222);
        transform: scale(1.02);
        box-shadow: 0px 4px 20px rgba(255,75,75,0.6);
    }

    /* Result card */
    .result-card {
        padding: 2.5em;
        margin: 2em auto;
        border-radius: 20px;
        text-align: center;
        font-size: 1.4em;
        font-weight: 600;
        box-shadow: 0px 6px 25px rgba(0,0,0,0.5);
        width: 60%;
    }
    .good {
        background: rgba(34,139,34,0.2);
        color: #90ee90;
        border: 2px solid #90ee90;
    }
    .bad {
        background: rgba(178,34,34,0.2);
        color: #ff7f7f;
        border: 2px solid #ff4b4b;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9em;
        margin: 3em 0 1em 0;
        color: #aaa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown("<h2>🍷 Premium Wine Quality Predictor</h2>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover if your wine meets premium quality standards</p>', unsafe_allow_html=True)

# --- Input Section (Glass Card) ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

fixed_acidity = st.slider("Fixed Acidity", 4.0, 15.0, 7.4)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.3)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.5)
residual_sugar = st.slider("Residual Sugar", 0.5, 10.0, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.07)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 30)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 100)
density = st.slider("Density", 0.990, 1.004, 0.995)
pH = st.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.75)
alcohol = st.slider("Alcohol", 8.0, 15.0, 12.5)

st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction ---
inputs = [[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, pH, sulphates, alcohol
]]

inputs_scaled = scaler.transform(inputs)
prediction = model.predict(inputs_scaled)[0]
probability = model.predict_proba(inputs_scaled)[0]

if st.button("🔮 Predict Wine Quality"):
    if prediction == 1:
        st.markdown(
            f'<div class="result-card good">🍇 Excellent Choice!<br>This wine is <b>Good Quality</b><br><br>Confidence: {probability[1]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-card bad">🍇 Not Quite There...<br>This wine is <b>Not Good Quality</b><br><br>Confidence: {probability[0]*100:.2f}%</div>',
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown('<p class="footer">Made with ❤️ for Mr. Sanborn • Powered by Streamlit</p>', unsafe_allow_html=True)
