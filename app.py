import streamlit as st
import pandas as pd
import joblib

# ------------------- LOAD MODEL + SCALER -------------------
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ------------------- STREAMLIT SETTINGS -------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ------------------- MAIN UI -------------------
st.title("🍷 Wine Quality Prediction Dashboard")
st.markdown("<h3 style='color:#4B0000;'>A refined tool for predicting premium wine quality</h3>", unsafe_allow_html=True)

st.markdown("""
Welcome to the **Wine Quality Prediction App**!  
This tool uses a **Random Forest Classifier** trained on cleaned and balanced wine data.  
Use the sidebar to set wine chemistry attributes and discover if your wine is of premium quality.  
""")

# ------------------- CUSTOM STYLE -------------------
st.markdown("""
    <style>
    /* Background */
    .main {
        background-color: #FDFDFD;  /* light grayish white */
    }
    /* Title */
    h1, h2, h3 {
        color: #2C3E50;
        font-family: 'Georgia', serif;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ECF0F1; /* light cool gray */
    }
    section[data-testid="stSidebar"] .stSlider label, 
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #2C3E50;
    }
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3498DB, #2980B9);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2980B9, #3498DB);
        color: #F1C40F;
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
        background-color: #E8F8F5;
        color: #145A32;
        border: 3px solid #27AE60;
    }
    .bad {
        background-color: #FDEDEC;
        color: #922B21;
        border: 3px solid #C0392B;
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

# ------------------- FEATURE NAMES -------------------
feature_names = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# ------------------- PREDICTION -------------------
if st.sidebar.button("🍇 Predict Quality"):
    input_df = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]], columns=feature_names)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

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
    st.dataframe(input_df, use_container_width=True)
