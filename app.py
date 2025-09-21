import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")

# --- Custom CSS for Modern Premium UI ---
st.markdown(
    """
    <style>
    /* Background gradient */
    .main, .block-container {
        background: linear-gradient(135deg, #2a0a0a 0%, #0d0d0d 100%);
        color: #f5f5f5;
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
    }
    p.subtitle {
        text-align: center;
        color: #ddd;
        font-size: 1.1em;
        margin-bottom: 2em;
    }

    /* Glassmorphic card */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 2em;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        backdrop-filter: blur(8px);
        margin-bottom: 2em;
    }

    /* Sliders */
    div[data-baseweb="slider"] > div {
        height: 10px !important;
        background: rgba(255,255,255,0.15);
        border-radius: 6px;
    }
    div[data-baseweb="slider"] span {
        height: 22px !important;
        width: 22px !important;
        background: #ff4b4b !important;
        border: 2px solid white !important;
        border-radius: 50%;
        box-shadow: 0px 0px 10px #ff4b4b;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #b22222, #ff4b4b);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 0.7em 1.4em;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff4b4b, #b22222);
        transform: scale(1.02);
        box-shadow: 0px 4px 15px rgba(255,75,75,0.5);
    }

    /* Result card */
    .result-card {
        padding: 2em;
        margin-top: 1.5em;
        border-radius: 18px;
        text-align: center;
        font-size: 1.3em;
        font-weight: 600;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.45);
    }
    .good {
        background: rgba(34,139,34,0.15);
        color: #90ee90;
        border: 2px solid #90ee90;
    }
    .bad {
        background: rgba(178,34,34,0.15);
        color: #ff7f7f;
        border: 2px solid #ff4b4b;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85em;
        margin-top: 2em;
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

