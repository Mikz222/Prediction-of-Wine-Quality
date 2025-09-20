import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("artifacts/model.pkl")

# Page config
st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

# Banner
st.title("🍷 Wine Quality Prediction Dashboard")
st.markdown("Predict wine quality based on its chemical composition.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dataset Info", "ℹ️ About"])

# ---------------- TAB 1: PREDICTION ----------------
with tab1:
    st.subheader("Enter Wine Measurements")

    # Layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4, help="Higher values = more tart taste")
        volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7, help="Too high = vinegar taste")
        citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.0, help="Adds freshness & flavor")
        residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 1.9, help="Sweetness level")
        chlorides = st.number_input("Chlorides", 0.0, 0.2, 0.076, help="Salt content")

    with col2:
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 80.0, 11.0, help="Preservative")
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0, help="Total preservative content")
        density = st.number_input("Density", 0.990, 1.005, 0.9978, help="Close to water = lighter wine")
        pH = st.number_input("pH", 2.5, 4.5, 3.3, help="Acidity level")
        sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56, help="Wine stability & flavor")
        alcohol = st.number_input("Alcohol", 8.0, 15.0, 9.4, help="Alcohol % content")

    # Prediction
    if st.button("🍇 Predict Quality", use_container_width=True):
        features = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                     chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                     density, pH, sulphates, alcohol]]
        
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.success(f"✅ This wine is predicted to be **Good Quality (≥7)** 🍷")
        else:
            st.error(f"❌ This wine is predicted to be **Not Good Quality (<7)**")

        st.metric(label="Confidence Level", value=f"{proba*100:.2f}%")

        # Show summary table of inputs
        st.write("📌 Entered Measurements:")
        df = pd.DataFrame(features, columns=[
            "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", 
            "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol"
        ])
        st.dataframe(df, use_container_width=True)

# ---------------- TAB 2: DATASET INFO ----------------
with tab2:
    st.subheader("📊 Wine Dataset Information")
    st.markdown("""
    The model was trained on the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality),
    which contains **red wine samples** with physicochemical measurements.

    - 🍇 **Inputs**: 11 chemical properties (acidity, alcohol, sulphates, etc.)
    - 🎯 **Target**: Wine quality (score 0–10, we classify ≥7 as good)
    - 📦 **Samples**: ~1600 wines
    """)

# ---------------- TAB 3: ABOUT ----------------
with tab3:
    st.subheader("ℹ️ About this App")
    st.markdown("""
    This app helps predict whether a wine is of **good quality** based on its 
    chemical composition using a **Random Forest Classifier**.

    - 🔬 Built with **scikit-learn & joblib**
    - 🖥️ Deployed with **Streamlit**
    - 👨‍💻 Created as part of a **Machine Learning project**

    ✨ Try changing the inputs to see how each chemical property affects wine quality!
    """)
