import streamlit as st
import numpy as np

st.set_page_config(page_title="Wine Quality App", layout="wide")


st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #ffdde1, #ee9ca7);
        }
        .stSelectbox label {
            font-size: 16px !important;
            color: #8e44ad !important;
            font-weight: 600;
        }
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 20px !important;
            border: 2px solid #f78fb3 !important;
            background-color: #fff0f6 !important;
            font-size: 15px !important;
        }
        .stButton>button {
            background: linear-gradient(to right, #ff6b81, #ff9ff3);
            color: white;
            border-radius: 30px;
            border: none;
            padding: 12px 25px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0px 4px 10px rgba(255, 0, 150, 0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #ff9ff3, #ff6b81);
            transform: scale(1.03);
        }
        .result-box {
            background-color: #ffe6f0;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: #6a0572;
            font-size: 20px;
            font-weight: bold;
            box-shadow: 0px 4px 12px rgba(200, 0, 100, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

st.title("🍷 Girly Wine Quality App 💖")

#  Two-column layout
col1, col2 = st.columns([5, 6])

with col1:
    fixed_acidity = st.selectbox("Fixed Acidity", [round(x,1) for x in np.arange(4.0, 16.1, 0.5)])
    volatile_acidity = st.selectbox("Volatile Acidity", [round(x,2) for x in np.arange(0.1, 1.6, 0.05)])
    citric_acid = st.selectbox("Citric Acid", [round(x,2) for x in np.arange(0.0, 1.1, 0.05)])
    residual_sugar = st.selectbox("Residual Sugar", [round(x,1) for x in np.arange(0.5, 15.1, 0.5)])
    chlorides = st.selectbox("Chlorides", [round(x,3) for x in np.arange(0.01, 0.21, 0.01)])
    free_sulfur_dioxide = st.selectbox("Free Sulfur Dioxide", list(range(1, 73)))

with col2:
    total_sulfur_dioxide = st.selectbox("Total Sulfur Dioxide", list(range(6, 290, 5)))
    density = st.selectbox("Density", [round(x,3) for x in np.arange(0.990, 1.006, 0.001)])
    pH = st.selectbox("pH", [round(x,2) for x in np.arange(2.5, 4.6, 0.05)])
    sulphates = st.selectbox("Sulphates", [round(x,2) for x in np.arange(0.3, 2.1, 0.05)])
    alcohol = st.selectbox("Alcohol %", [round(x,1) for x in np.arange(8.0, 15.1, 0.5)])

# Predict Button
if st.button("💖 Predict Wine Quality 💖"):
    st.markdown('<div class="result-box">✨ Your wine is likely to be of GOOD QUALITY! 🍷💎</div>', unsafe_allow_html=True)
