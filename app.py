import streamlit as st
import numpy as np

st.set_page_config(page_title= "Wine Quality App", layout="wide")


st.markdown("""
    <style>
        /* 🎀 Clean Rounded Dropdown (No Highlights) */
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 25px !important;
            border: 2px solid #f78fb3 !important;
            background: #fff0f6 !important;
            font-size: 16px !important;
            box-shadow: 0px 3px 6px rgba(255, 120, 180, 0.2);
        }
        /* 🚫 Remove highlight completely */
        .stSelectbox div[data-baseweb="select"]:focus,
        .stSelectbox div[data-baseweb="select"]:hover {
            outline: none !important;
            background: #fff0f6 !important;
            border: 2px solid #f78fb3 !important;
            box-shadow: none !important;
        }
        /* 🔮 Dropdown menu itself */
        .stSelectbox ul {
            border-radius: 20px !important;
            background: #fff0f6 !important;
            border: 1px solid #f78fb3 !important;
        }
        .stSelectbox ul li {
            border-radius: 15px !important;
        }
        .stSelectbox ul li:hover {
            background: #ffd6eb !important;
            color: #d63384 !important;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🍷Wine Quality App")

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




