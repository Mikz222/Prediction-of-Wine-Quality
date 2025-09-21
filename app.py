import streamlit as st
import joblib

# Load artifacts
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Force fullscreen dark background */
    .stApp {
        background: linear-gradient(160deg, #0a0a0a, #1a1a1a 90%) !important;
        color: #f0f0f0 !important;
    }
    .main {
        background: transparent !important; /* remove white layer */
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 3em 1em 2em 1em;
    }
    .hero h1 {
        font-size: 3.2em;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ffcccc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2em;
    }
    .hero p {
        color: #bbb;
        font-size: 1.1em;
        margin-top: 0;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 2em;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.9);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: scale(1.01);
    }

    /* Sliders */
    div[data-baseweb="slider"] > div {
        height: 14px !important;
        background: rgba(255,255,255,0.15);
        border-radius: 12px;
    }
    div[data-baseweb="slider"] span {
        height: 28px !important;
        width: 28px !important;
        background: #ff4b4b !important;
        border: 3px solid white !important;
        border-radius: 50%;
        box-shadow: 0px 0px 15px #ff4b4b;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #b22222);
        color: white;
        font-size: 20px;
        font-weight: 700;
        padding: 0.9em 1.6em;
        border-radius: 14px;
        border: none;
        width: 100%;
        transition: all 0.25s ease-in-out;
        box-shadow: 0px 4px 20px rgba(255,75,75,0.5);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 25px rgba(255,75,75,0.7);
    }

    /* Result card */
    .result-card {
        padding: 2em;
        margin: 2em auto;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5em;
        font-weight: 700;
        width: 70%;
        animation: fadeIn 0.8s ease-in-out;
    }
    .good {
        background: rgba(0, 128, 0, 0.15);
        color: #90ee90;
        border: 2px solid #32cd32;
        box-shadow: 0px 0px 25px rgba(50,205,50,0.6);
    }
    .bad {
        background: rgba(178,34,34,0.15);
        color: #ff7f7f;
        border: 2px solid #ff4b4b;
        box-shadow: 0px 0px 25px rgba(255,75,75,0.6);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(25px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9em;
        margin: 3em 0 1em 0;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Hero Header ---
st.markdown(
    """
    <div class="hero">
        <h1>🍷 Premium Wine Quality Predictor</h1>
        <p>Discover if your wine meets the <b>premium standard</b>. Adjust the chemistry sliders and test the blend.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Layout: Inputs Left, Wine Glass Right ---
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
            <h3>Wine Glass Profile</h3>
            <p style="color:#ccc;">Your chemistry settings reflect this pour:</p>
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
            f'<div class="result-card good">✨ Premium Wine Detected!<br>This wine is <b>Good Quality</b> 🍷<br><br>Confidence: {probability[1]*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-card bad">⚠️ Needs Refinement...<br>This wine is <b>Not Good Quality</b> 🍷<br><br>Confidence: {probability[0]*100:.2f}%</div>',
            unsafe_allow_html=True
        )





