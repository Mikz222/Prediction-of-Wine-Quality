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
        .top-bar {
            background-color: #1877f2;
            color: white;
            padding: 18px;
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            border-radius: 0 0 10px 10px;
            margin-bottom: 30px;
        }
        h2 {
            font-size: 28px !important;
            text-align: center;
            margin-bottom: 8px;
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
            border-radius: 8px;
            height: 50px;
            width: 100%;
            font-size: 18px;
            font-weight: 600;
            border: none;
        }
        .stButton>button:hover {
            background-color: #166fe5;
        }
        /* Result Cards */
        .result-card {
            padding: 30px;
            margin-top: 30px;
            border-radius: 14px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            animation: fadeIn 0.6s ease-in-out;
        }
        .good {
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
            color: #1b5e20;
            border: 2px solid #2e7d32;
        }
        .bad {
            background: linear-gradient(135deg, #ffebee, #ffcdd2);
            color: #b71c1c;
            border: 2px solid #d32f2f;
        }
        .result-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .confidence {
            font-size: 18px;
            color: #333;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(15px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# ================== Top Navbar ==================
st.markdown("<div class='top-bar'>🍷 Wine Quality Predictor</div>", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h2>Predict Your Wine’s Quality</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Select the attributes below to check if your wine is good quality.</p>", unsafe_allow_html=True)

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
        st.markdown(
            f"""
            <div class='result-card good'>
                <div class='result-title'>🍷 Good Quality Wine</div>
                <div class='confidence'>Confidence: {probability[1]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class='result-card bad'>
                <div class='result-title'>🚫 Not Good Quality Wine</div>
                <div class='confidence'>Confidence: {probability[0]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True
        )
