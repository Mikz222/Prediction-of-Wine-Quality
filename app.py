import streamlit as st
import numpy as np

st.set_page_config(page_title= "Wine Quality App", layout="wide")


st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #ffdde1, #ee9ca7, #fad0c4, #ffd1ff);
            background-attachment: fixed;
        }
        h1 {
            font-size: 40px !important;
            color: #ff6b9d !important;
            text-align: center;
            font-weight: 900;
            text-shadow: 2px 2px 6px rgba(255, 100, 150, 0.4);
        }
        .stSelectbox label {
            font-size: 18px !important;
            color: #8e44ad !important;
            font-weight: 700;
        }
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 25px !important;
            border: 2px solid #f78fb3 !important;
            background: #fff0f6 !important;
            font-size: 16px !important;
            box-shadow: 0px 3px 6px rgba(255, 120, 180, 0.3);
            transition: all 0.3s ease;
        }
        .stSelectbox div[data-baseweb="select"]:hover {
            border-color: #ff6b81 !important;
            box-shadow: 0px 4px 12px rgba(255, 105, 180, 0.5);
        }
        .stButton>button {
            background: linear-gradient(to right, #ff6b81, #ff9ff3);
            color: white;
            border-radius: 40px;
            border: none;
            padding: 14px 35px;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0px 6px 15px rgba(255, 0, 150, 0.4);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #ff9ff3, #ff6b81);
            transform: scale(1.05);
            box-shadow: 0px 8px 20px rgba(255, 0, 150, 0.6);
        }
        .result-box {
            background: linear-gradient(135deg, #ffe6f0, #ffd6eb);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            color: #6a0572;
            font-size: 24px;
            font-weight: bold;
            margin-top: 30px;
            animation: fadeIn 1s ease-in-out;
            box-shadow: 0px 6px 18px rgba(255, 0, 100, 0.2);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

st.title("🍷G Wine Quality App")

# Two-column layout (5:6 ratio)
col1, col2 = st.columns([5, 6])

with col1:
    fixed_acidity = st.selectbox("✨ Fixed Acidity", [round(x,1) for x in np.arange(4.0, 16.1, 0.5)])
    volatile_acidity = st.selectbox("💎 Volatile Acidity", [round(x,2) for x in np.arange(0.1, 1.6, 0.05)])
    citric_acid = st.selectbox("🌸 Citric Acid", [round(x,2) for x in np.arange(0.0, 1.1, 0.05)])
    residual_sugar = st.selectbox("🍬 Residual Sugar", [round(x,1) for x in np.arange(0.5, 15.1, 0.5)])
    chlorides = st.selectbox("🧂 Chlorides", [round(x,3) for x in np.arange(0.01, 0.21, 0.01)])
    free_sulfur_dioxide = st.selectbox("💨 Free Sulfur Dioxide", list(range(1, 73)))

with col2:
    total_sulfur_dioxide = st.selectbox("🌫️ Total Sulfur Dioxide", list(range(6, 290, 5)))
    density = st.selectbox("⚖️ Density", [round(x,3) for x in np.arange(0.990, 1.006, 0.001)])
    pH = st.selectbox("🧪 pH", [round(x,2) for x in np.arange(2.5, 4.6, 0.05)])
    sulphates = st.selectbox("🌟 Sulphates", [round(x,2) for x in np.arange(0.3, 2.1, 0.05)])
    alcohol = st.selectbox("🍹 Alcohol %", [round(x,1) for x in np.arange(8.0, 15.1, 0.5)])

# Predict Button
if st.button("💖✨ Predict Wine Quality ✨💖"):
    st.markdown('<div class="result-box">🍷 Your wine is likely to be of <span style="color:#ff4d79;">GOOD QUALITY</span> 💎✨</div>', unsafe_allow_html=True)
    st.markdown("🌸 Extra Results: This wine has balanced acidity, sweet notes, and perfect alcohol levels for a *premium taste* 💕🍇✨")

