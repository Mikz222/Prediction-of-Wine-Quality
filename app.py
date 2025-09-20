import streamlit as st
import pandas as pd
import pickle

# Load model + scaler
model = pickle.load(open("artifacts/model.pkl", "rb"))
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
# Page setup
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

# Custom CSS for light clean design
st.markdown("""
    <style>
    body {
        background-color: #EAF4FC;
        color: #2E4057;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput, .stNumberInput {
        border-radius: 10px;
        background-color: #FFFFFF !important;
        padding: 8px;
    }
    .prediction-box {
        border: 2px solid #B0C4DE;
        border-radius: 15px;
        background-color: #F8FBFF;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .good {
        color: #2E8B57;
        font-weight: bold;
        font-size: 22px;
    }
    .bad {
        color: #B22222;
        font-weight: bold;
        font-size: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)

# Input fields (two-column layout like your sketch)
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)

with col2:
    chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
    sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
    alcohol = st.number_input("Alcohol %", min_value=0.0, step=0.1)
    pH = st.number_input("pH", min_value=0.0, step=0.01)

# Prediction button
if st.button("Predict Quality"):
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                   chlorides, sulphates, alcohol, pH]]
    input_df = pd.DataFrame(input_data, 
        columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "sulphates", "alcohol", "pH"])
    
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    # Show result in a nice styled box
    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown("<p class='good'>✅ Good Quality Wine</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='bad'>❌ Poor Quality Wine</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

