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
        .stApp {
            background-color: #f0f2f5;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #1c1e21;
        }
        .main-card {
            max-width: 600px;
            margin: 40px auto;
            padding: 40px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            font-size: 34px !important;
            text-align: center;
            color: #1877f2 !important;
            margin-bottom: 10px;
        }
        p.subtitle {
            text-align: center;
            font-size: 16px;
            color: #606770;
            margin-bottom: 25px;
        }
        .stSelectbox label {
            font-weight: 500 !important;
            color: #050505 !important;
        }
        .stButton>button {
            background-color: #1877f2;
            color: white;
            border-radius: 6px;
            height: 48px;
            width: 100%;
            font-size: 16px;
            font-weight: 600;
            border: none;
        }
        .stButton>button:hover {
            background-color: #166fe5;
        }
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


st.markdown("<h1>Wine Quality Predictor 🍷</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Select the values below and see if your wine passes the test</p>", unsafe_allow_html=True)

# ================== Dropdown Inputs ==================
fixed_acidity = st.selectbox("Fixed Acidity", [round(x,1) for x in np.arange(4.0, 16.1, 0.5)])
volatile_acidity = st.selectbox("Volatile Acidity", [round(x,2) for x in np.arange(0.1, 1.6, 0.05)])
citric_acid = st.selectbox("Citric Acid", [round(x,2) for x in np.arange(0.0, 1.1, 0.05)])
residual_sugar = st.selectbox("Residual Sugar", [round(x,1) for x in np.arange(0.5, 15.1, 0.5)])
chlorides = st.selectbox("Chlorides", [round(x,3) for x in np.arange(0.01, 0.21, 0.01)])
free_sulfur_dioxide = st.selectbox("Free Sulfur Dioxide", list(range(1, 73)))
total_sulfur_dioxide = st.selectbox("Total Sulfur Dioxide", list(range(6, 290, 5)))
density = st.selectbox("Density", [round(x,3) for x in np.arange(0.990, 1.006, 0.001)])
pH = st.selectbox("pH", [round(x,2) for x in np.arange(2.5, 4.6, 0.05)])
sulphates = st.selectbox("Sulphates", [round(x,2) for x in np.arange(0.3, 2.1, 0.05)])
alcohol = st.selectbox("Alcohol %", [round(x,1) for x in np.arange(8.0, 15.1, 0.5)])

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

