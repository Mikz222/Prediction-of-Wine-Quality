import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    /* Center main content */
    .block-container {
        max-width: 700px;
        padding-top: 2rem;
        margin: auto;
    }

    /* Thicker slider track */
    div[data-baseweb="slider"] > div {
        height: 8px !important;
    }

    /* Slider handle */
    div[data-baseweb="slider"] span {
        height: 22px !important;
        width: 22px !important;
        background: #8B0000 !important;  /* dark red */
        border: 2px solid #fff !important;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.3);
    }

    /* Slider track color */
    div[data-baseweb="slider"] div[role="slider"]::before {
        background: #8B0000 !important;
    }

    /* Title */
    h2 {
        color: #8B0000;
        text-align: center;
        font-weight: 700;
    }

    /* Subtitle */
    p.subtitle {
        text-align: center;
        color: gray;
        font-size: 1.1em;
    }

    /* Predict button */
    .stButton > button {
        background: #8B0000;
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 0.6em 1.2em;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: #A52A2A;
        transform: scale(1.02);
    }

    /* Result card */
    .result-card {
        padding: 1.5em;
        margin-top: 1.5em;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .good {
        background: #e8f5e9;
        color: #1b5e20;
        border: 2px solid #43a047;
    }
    .bad {
        background: #ffebee;
        color: #b71c1c;
        border: 2px solid #e53935;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown("<h2>🍷 Wine Quality Prediction</h2>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Adjust the sliders below and predict your wine’s quality</p>', unsafe_allow_html=True)

# --- Sliders ---
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

# --- Collect input ---
inputs = [[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, pH, sulphates, alcohol
]]

inputs_scaled = scaler.transform(inputs)
prediction = model.predict(inputs_scaled)[0]
probability = model.predict_proba(inputs_scaled)[0]

# --- Button & Result ---
if st.button("🔮 Predict Quality"):
    if prediction == 1:
        st.markdown(
            f'<div class="result-card good">✅ This wine is predicted to be <b>Good Quality</b><br>Confidence: {probability[1]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-card bad">❌ This wine is predicted to be <b>Not Good Quality</b><br>Confidence: {probability[0]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
