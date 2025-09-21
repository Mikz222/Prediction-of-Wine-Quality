import streamlit as st
import joblib
import numpy as np

# ================== Page Config ==================
st.set_page_config(
    page_title="Wine Quality Predictor 🍷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================== Load Model & Scaler ==================
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        /* Base page */
        .stApp {
            background-color: #f0f2f5; /* FB style light gray */
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #1c1e21;
        }

        /* Centered white card */
        .main-card {
            max-width: 600px;
            margin: 40px auto;
            padding: 40px 40px 50px 40px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        }

        /* Header */
        h1 {
            font-size: 36px !important;
            text-align: center;
            color: #1877f2 !important; /* FB blue */
            margin-bottom: 10px;
        }
        p.subtitle {
            text-align: center;
            font-size: 16px;
            color: #606770;
            margin-bottom: 25px;
        }

        /* Sliders */
        .stSlider label {
            font-size: 15px !important;
            font-weight: 500 !important;
            color: #050505 !important;
        }
        div[data-baseweb="slider"] > div {
            background: #ddd !important;
            height: 6px;
            border-radius: 6px;
        }
        div[data-baseweb="slider"] span {
            background: #1877f2 !important;
            border: none;
            height: 18px;
            width: 18px;
            border-radius: 50%;
        }

        /* Button */
        .stButton>button {
            background-color: #1877f2;
            color: white;
            border-radius: 6px;
            height: 48px;
            width: 100%;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #166fe5;
        }

        /* Result Cards */
        .result-card {
            padding: 18px;
            margin-top: 25px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
        }
        .good {
            background: #e7f3e7;
            color: #2e7d32;
            border: 1px solid #2e7d32;
        }
        .bad {
            background: #fbeaea;
            color: #d32f2f;
            border: 1px solid #d32f2f;
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Layout ==================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<h1>Wine Quality Predictor 🍷</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Adjust the sliders below and see if your wine passes the test</p>", unsafe_allow_html=True)

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
if st.button("🔮 Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.markdown(f"<div class='result-card good'>✅ This wine is predicted to be Good Quality<br>Confidence: {probability[1]*100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-card bad'>❌ This wine is predicted to be Not Good Quality<br>Confidence: {probability[0]*100:.2f}%</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
