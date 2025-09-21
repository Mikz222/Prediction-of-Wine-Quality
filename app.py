import streamlit as st
import joblib

# Load artifacts
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

st.markdown(
    """
    <style>
    /* Old Skool Vintage Background */
    .stApp {
        background: radial-gradient(circle, #3b2f2f 0%, #1a0d0d 100%) !important;
        color: #f5f2e7 !important;
        font-family: 'Georgia', serif !important;
        font-size: 18px;
    }
    .main {
        background: transparent !important;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.5em 1em 1.5em 1em;
        border-bottom: 3px double #a67c52;
    }
    .hero h1 {
        font-size: 3.5em !important;
        font-weight: bold !important;
        color: #d4af37;
        text-shadow: 2px 2px 6px #000;
        margin-bottom: 0.3em;
    }
    .hero p {
        color: #e6decf !important;
        font-size: 1.2em !important;
        font-style: italic;
    }

    /* Vintage card */
    .glass-card {
        background: rgba(56, 36, 28, 0.85);
        border: 2px solid #a67c52;
        border-radius: 10px;
        padding: 2em;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.7);
        font-size: 1.1em !important;
        margin-bottom: 1.5em;
    }

    /* Sliders */
    div[data-baseweb="slider"] > div {
        height: 12px !important;
        background: #5a3d2b !important;
        border-radius: 10px;
    }
    div[data-baseweb="slider"] span {
        height: 26px !important;
        width: 26px !important;
        background: #d4af37 !important;
        border: 2px solid #fff5d7 !important;
        border-radius: 50%;
        box-shadow: 0px 0px 10px #d4af37;
    }
    label {
        font-size: 1.1em !important;
        font-weight: 600 !important;
        color: #f5f2e7 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #6b4226, #3b1f0e);
        color: #f5f2e7 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        padding: 0.9em 1.5em;
        border-radius: 8px;
        border: 2px solid #d4af37;
        width: 100%;
        transition: all 0.25s ease-in-out;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.8);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #3b1f0e, #6b4226);
        transform: scale(1.04);
        box-shadow: 0px 6px 20px rgba(212,175,55,0.7);
    }

    /* Result card */
    .result-card {
        padding: 2em;
        margin: 2em auto;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6em !important;
        font-weight: bold !important;
        width: 75%;
        border: 2px solid #d4af37;
        background: rgba(30, 15, 10, 0.9);
        color: #f5f2e7;
    }
    .good {
        border-color: #4caf50;
        color: #90ee90 !important;
    }
    .bad {
        border-color: #ff4b4b;
        color: #ff9999 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 1em !important;
        margin: 3em 0 1em 0;
        color: #d4af37 !important;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Hero Header ---
st.markdown(
    """
    <div class="hero">
        <h1>🍷 Old Skool Wine Predictor</h1>
        <p>A retro touch to modern machine learning predictions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Layout ---
col1, col2 = st.columns([2,1], gap="large")

with col1:
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

with col2:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <h3 style="color:#d4af37;">Vintage Wine Glass</h3>
            <p style="color:#e6decf;">Your settings reflect this classic pour:</p>
            <img src="https://cdn-icons-png.flaticon.com/512/931/931949.png" width="120">
        </div>
        """,
        unsafe_allow_html=True
    )

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
            f'<div class="result-card good">🍇 Premium Vintage Detected!<br>This wine is <b>Good Quality</b> 🍷<br><br>Confidence: {probability[1]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-card bad">⚠️ Cellar Needs Refinement...<br>This wine is <b>Not Good Quality</b> 🍷<br><br>Confidence: {probability[0]*100:.2f}%</div>',
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown('<p class="footer">🍷 Crafted in Retro Style • Vintage Edition</p>', unsafe_allow_html=True)
