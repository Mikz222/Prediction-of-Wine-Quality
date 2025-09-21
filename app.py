import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")

# --- Custom CSS for premium UI ---
st.markdown(
    """
    <style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Center content */
    .block-container {
        max-width: 750px;
        padding-top: 2rem;
        margin: auto;
    }

    /* Title */
    h2 {
        color: #8B0000;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.3em;
    }
    p.subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 2em;
    }

    /* Slider design */
    div[data-baseweb="slider"] > div {
        height: 10px !important;
    }
    div[data-baseweb="slider"] span {
        height: 24px !important;
        width: 24px !important;
        background: linear-gradient(135deg, #8B0000, #A52A2A) !important;
        border: 2px solid #fff !important;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.3);
    }
    div[data-baseweb="slider"] div[role="slider"]::before {
        background: #8B0000 !important;
    }

    /* Input section card */
    .input-card {
        background: #fff;
        padding: 2em;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 2em;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #8B0000, #A52A2A);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 0.7em 1.4em;
        border-radius: 12px;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #A52A2A, #8B0000);
        transform: translateY(-2px);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
    }

    /* Result card */
    .result-card {
        padding: 1.8em;
        margin-top: 1.5em;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2em;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.12);
    }
    .good {
        background: #f1f8f4;
        color: #1b5e20;
        border-left: 6px solid #43a047;
    }
    .bad {
        background: #fcebea;
        color: #b71c1c;
        border-left: 6px solid #e53935;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85em;
        margin-top: 2em;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown("<h2>🍷 Wine Quality Prediction</h2>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Adjust the sliders below and see if your wine is Good or Not Good</p>', unsafe_allow_html=True)

# --- Input Section in Card ---
st.markdown('<div class="input-card">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)  # close card

# --- Collect input ---
inputs = [[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, pH, sulphates, alcohol
]]

inputs_scaled = scaler.transform(inputs)
prediction = model.predict(inputs_scaled)[0]
probability = model.predict_proba(inputs_scaled)[0]

# --- Prediction Button ---
if st.button("🔮 Predict Wine Quality"):
    if prediction == 1:
        st.markdown(
            f'<div class="result-card good">✅ This wine is predicted to be <b>Good Quality</b><br><br>Confidence: {probability[1]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-card bad">❌ This wine is predicted to be <b>Not Good Quality</b><br><br>Confidence: {probability[0]*100:.2f}%</div>',
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown('<p class="footer">Made with ❤️ and Streamlit</p>', unsafe_allow_html=True)
