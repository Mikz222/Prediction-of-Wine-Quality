import streamlit as st
import joblib

# Load trained model and scaler
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.title("🍷 Wine Quality Prediction")

st.write("Enter the wine’s chemical measurements below:")

# Input fields for all features
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 1.9)
chlorides = st.number_input("Chlorides", 0.0, 0.2, 0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 80.0, 11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.number_input("Density", 0.990, 1.005, 0.9978)
pH = st.number_input("pH", 2.5, 4.5, 3.3)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

# Predict button
if st.button("Predict Quality"):
    features = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]]

    # Scale input features using same scaler from training
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    # Interpret result
    if prediction == 1:
        st.success("✅ This wine is predicted to be GOOD quality (≥7)")
    else:
        st.error("❌ This wine is predicted to be NOT good quality (<7)")
