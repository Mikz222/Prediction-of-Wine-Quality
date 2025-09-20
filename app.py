# ------------------- PREDICTION -------------------
if st.sidebar.button("🍇 Predict Quality"):
    features = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]]
    
    # Apply scaler before prediction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]  # [prob_not_good, prob_good]

    # Labels
    class_labels = {0: "Not Good Quality", 1: "Good Quality"}
    confidence = round(probs[prediction] * 100, 2)  # confidence of predicted class

    # Result card with confidence under the text
    if prediction == 1:
        st.markdown(
            f"""
            <div class='result-card good'>
                ✅ Excellent! This wine is predicted to be <br>
                <span style='font-size:28px;'>{class_labels[prediction]} 🍷</span><br>
                <span style='font-size:20px;'>Confidence: {confidence}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class='result-card bad'>
                ❌ Unfortunately, this wine is predicted to be <br>
                <span style='font-size:28px;'>{class_labels[prediction]}</span><br>
                <span style='font-size:20px;'>Confidence: {confidence}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Show entered values
    st.markdown("### 📌 Your Entered Measurements")
    df = pd.DataFrame(features, columns=[
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", 
        "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol"
    ])
    st.dataframe(df, use_container_width=True)
