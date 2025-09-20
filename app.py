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
    layout="centered"
)

# ------------------- CUSTOM STYLE -------------------
st.markdown("""
    <style>
    body {
        background-color: #F8FAFC; /* very light cool background */
        font-family: 'Trebuchet MS', sans-serif;
    }
    h1, h2, h3 {
        color: #1E3A8A; /* dark blue */
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3B82F6, #2563EB);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563EB, #1D4ED8);
        color: #FACC15;
    }
    .input-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
    }
    .good {
        background-color: #ECFDF5;
        color: #065F46;
        border: 3px solid #10B981;
    }
    .bad {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 3px solid #DC2626;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- MAIN UI -------------------
st.title("🍷 Wine Quality Prediction")
st.markdown("<h3 style='color:#1E3A8A;'>Predict wine quality with chemistry attributes</h3>", unsafe_allow_html=True)

st.write("Fill in the wine’s chemical measurements below and check whether it is **Good Quality** or **Not Good Quality**.")

# ------------------- INPUT FORM -------------------
with st.form("wine_form"):
    st.markdown("### 🔹 Enter Wine Attributes")

    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4)
        volatile_acidity = st.number_input("Volatile Acidity", 0.0, 1.5, 0.7)
        citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
        residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 1.9)
        chlorides = st.number_input("Chlorides", 0.01, 0.2, 0.076)

    with col2:
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 80.0, 11.0)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
        density = st.number_input("Density", 0.990, 1.005, 0.9978, format="%.4f")
        pH = st.number_input("pH", 2.5, 4.5, 3.3)
        sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
        alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4)

    submitted = st.form_submit_button("🍇 Predict Quality")

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
if submitted:
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

    # Show result
    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>✅ Good Quality Wine<br><br>Confidence: {proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>❌ Not Good Quality Wine<br><br>Confidence: {(1-proba)*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    # Show entered values
    st.markdown("### 📊 Your Entered Measurements")
    st.dataframe(input_df, use_container_width=True)
