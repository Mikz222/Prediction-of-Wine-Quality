import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")

# Title
st.markdown(
    """
    <h2 style="text-align:center; color:#8B0000;">
        🍷 Wine Quality Prediction
    </h2>
    <p style="text-align:center; color:gray;">
        Adjust the sliders below and see if your wine is Good or Not Good.
    </p>
    """,
    unsafe_allow_html=True
)

# Sliders in main page (centered)
st.markdown("<br>", unsafe_allow_html=True)  # spacing

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

# Collect input
inputs = [[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, pH, sulphates, alcohol
]]

# Scale input
inputs_scaled = scaler.transform(inputs)

# Predict
prediction = model.predict(inputs_scaled)[0]
probability = model.predict_proba(inputs_scaled)[0]

# Button for prediction
if st.button("🔮 Predict Quality", use_container_width=True):
    if prediction == 1:
        st.success(f"✅ This wine is predicted to be **Good Quality** \n\n Confidence: {probability[1]*100:.2f}%")
    else:
        st.error(f"❌ This wine is predicted to be **Not Good Quality** \n\n Confidence: {probability[0]*100:.2f}%")
