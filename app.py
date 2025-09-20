import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("artifacts/model.pkl")

# Streamlit page settings
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ------------------- CUSTOM STYLE -------------------
st.markdown("""
    <style>
    /* Background */
    .main {
        background-color: #FDF6F0;
    }
    /* Title */
    h1, h2, h3 {
        color: #4B0000;
        font-family: 'Georgia', serif;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2E1A1A;
    }
    section[data-testid="stSidebar"] .stSlider label, 
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #FFD700;
    }
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #8B0000, #B22222);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #B22222, #8B0000);
        color: #FFD700;
    }
    /* Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        font-size: 22px;
        font-weight: bold;
        font-family: 'Trebuchet MS', sans-serif;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .good {
        background-color: #FFF9E6;
        color: #155724;
        border: 3px solid #FFD700;
    }
    .bad {
        background-color: #FCE8E6;
        color: #721c24;
        border: 3px solid #B22222;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("⚙️ Input Wine Measurements")
st.sidebar.markdown("✨ Use the sliders to adjust the wine attributes")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 1.5, 0.7)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.0, 15.0, 1.9)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.076)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 0.0, 80.0, 11.0)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.sidebar.slider("Density", 0.990, 1.005, 0.9978)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.sidebar.slider("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 9.4)

# ------------------- MAIN UI -------------------
st.title("🍷 Wine Quality Prediction Dashboard")
st.markdown("<h3 style='color:#8B0000;'>A refined tool for predicting premium wine quality</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("""
    Welcome to the **Wine Quality Prediction App**!  
    This tool uses a **Random Forest Classifier** trained on real-world data.  
    Use the sidebar to set wine chemistry attributes and discover if your wine is of premium quality.  
    """)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/184/184540.png", width=160)

# ------------------- PREDICTION -------------------
if st.sidebar.button("🍇 Predict Quality"):
    features = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                 chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                 density, pH, sulphates, alcohol]]
    
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    # Result card
    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>✅ Excellent! This wine is predicted to be <br><span style='font-size:28px;'>Good Quality 🍷</span><br>Confidence: {proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>❌ Unfortunately, this wine is predicted to be <br><span style='font-size:28px;'>Not Good Quality</span><br>Confidence: {(1-proba)*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    # Show entered values
    st.markdown("### 📌 Your Entered Measurements")
    df = pd.DataFrame(features, columns=[
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", 
        "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol"
    ])
    st.dataframe(df, use_container_width=True)


