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
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #8B0000;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #B22222;
        color: #fff;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        font-size: 20px;
        font-weight: bold;
    }
    .good {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #155724;
    }
    .bad {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("⚙️ Input Wine Measurements")
st.sidebar.markdown("Adjust the sliders to set wine attributes:")

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
st.markdown("A machine learning app that predicts if a wine is **good quality (≥7)** or **not good (<7)**.")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📊 Overview")
    st.markdown("""
    This tool uses a **Random Forest Classifier** trained on the 
    [UCI Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality).

    Adjust the inputs in the **sidebar** and click Predict to test different wines.
    """)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/184/184540.png", width=150)

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
            f"<div class='result-card good'>✅ Good Quality Wine<br>Confidence: {proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>❌ Not Good Quality Wine<br>Confidence: {(1-proba)*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    # Show entered values
    st.markdown("### 📌 Your Entered Measurements")
    df = pd.DataFrame(features, columns=[
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", 
        "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol"
    ])
    st.dataframe(df, use_container_width=True)

# ------------------- EXTRA SECTION -------------------
st.markdown("---")
st.subheader("ℹ️ About This Project")
st.markdown("""
- 🧪 Uses **11 wine chemistry features**  
- 🌲 Model: **Random Forest Classifier**  
- 🚀 Built with **Python, Scikit-learn, and Streamlit**  
- 📦 Deployed from GitHub  

Try adjusting the **alcohol** and **sulphates** sliders — these are among the most influential features for predicting quality!
""")
