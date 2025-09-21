import streamlit as st
import joblib
import numpy as np

# ================== Page Config ==================
st.set_page_config(
    page_title="💖 Wine Quality Predictor",
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
            background: linear-gradient(135deg, #ffdde1, #ee9ca7, #ffb6c1);
            color: #4a0033;
            font-family: 'Poppins', sans-serif;
        }

        /* Container */
        .block-container {
            max-width: 800px;
            margin: auto;
            padding-top: 40px;
        }

        /* Header */
        h1 {
            font-size: 3em;
            text-align: center;
            font-weight: 900;
            background: linear-gradient(90deg, #ff69b4, #ff1493, #ff6ec7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p {
            text-align: center;
            font-size: 1.2em;
            color: #660033;
            font-style: italic;
        }

        /* Sliders */
        .stSlider > div > div {
            background: #ffe4ec !important;
            height: 12px;
            border-radius: 10px;
        }
        div[data-baseweb="slider"] span {
            background: #ff69b4 !important;
            border: 3px solid white !important;
            height: 26px !important;
            width: 26px !important;
            border-radius: 50%;
            box-shadow: 0px 0px 15px rgba(255, 105, 180, 0.7);
        }
        label {
            font-size: 1.1em !important;
            color: #550033 !important;
            font-weight: 600 !important;
        }

        /* Button */
        .stButton>button {
            background: linear-gradient(90deg, #ff69b4, #ff1493, #ff6ec7);
            color: white !important;
            font-size: 20px !important;
            font-weight: 700 !important;
            padding: 1em 1.5em;
            border-radius: 30px;
            border: none;
            width: 100%;
            box-shadow: 0px 5px 25px rgba(255,105,180,0.7);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.05) rotate(-1deg);
            box-shadow: 0px 10px 35px rgba(255,20,147,0.9);
        }

        /* Result card */
        .result-card {
            margin-top: 30px;
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            font-size: 1.6em;
            font-weight: bold;
            font-family: 'Poppins', sans-serif;
        }
        .good {
            background: rgba(255,182,193,0.4);
            border: 3px solid #ff69b4;
            color: #b30059;
            box-shadow: 0px 0px 25px rgba(255,105,180,0.6);
        }
        .bad {
            background: rgba(255,228,225,0.6);
            border: 3px solid #ff1493;
            color: #800040;
            box-shadow: 0px 0px 25px rgba(255,20,147,0.6);
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h1>💖 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p>Slide with style, girl! Let’s see if your wine is fabulous 🍷✨</p>", unsafe_allow_html=True)

# ================== Sliders ==================
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

# ================== Prediction ==================
if st.button("💅✨ Predict My Wine ✨💅"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>🌸✨ Premium Babe Wine! ✨🌸<br>Good Quality 💖🍷<br>Confidence: {probability[1]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>🙅‍♀️ Not Slaying Yet...<br>Needs a Glow-Up 💔🍷<br>Confidence: {probability[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
