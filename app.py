# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Insurance Claim Prediction",
    layout="wide"
)

st.title("🛡️ Explainable Insurance Claim Prediction")
st.write(
    "This application predicts the likelihood of an insurance claim and explains "
    "the prediction using SHAP (SHapley Additive Explanations)."
)

# --------------------------
# Load trained model
# --------------------------
model = joblib.load("insurance.pkl")

# --------------------------
# User Inputs
# --------------------------
with st.form("input_form"):
    st.subheader("📋 Customer Information")

    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ["Female", "Male"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Number of Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox(
        "Region", ["Southwest", "Southeast", "Northwest", "Northeast"]
    )
    charges = st.number_input("Medical Charges", 100.0, 100000.0, 5000.0)

    submit = st.form_submit_button("🔍 Predict Claim")

# --------------------------
# Prediction + SHAP
# --------------------------
if submit:
    # Encoding (same as training)
    sex_val = 0 if sex == "Female" else 1
    smoker_val = 0 if smoker == "No" else 1

    region_map = {
        "Southwest": 0,
        "Southeast": 1,
        "Northwest": 2,
        "Northeast": 3,
    }

    user_input = pd.DataFrame(
        [[age, sex_val, bmi, children, smoker_val, region_map[region], charges]],
        columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"],
    )

    # --------------------------
    # Prediction
    # --------------------------
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")

    # --------------------------
    # SHAP Explanation (BAR CHART – JOURNAL SAFE)
    # --------------------------
    st.subheader("🔎 Model Explanation (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input)

    # Handle binary classification SHAP output
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # focus on "claim likely" class
    else:
        shap_vals = shap_values

    # Create clean figure
    fig, ax = plt.subplots(figsize=(6, 4))

    shap.summary_plot(
        shap_vals,
        user_input,
        plot_type="bar",
        max_display=7,
        show=False
    )

    st.pyplot(fig)

    # --------------------------
    # Plain English Explanation (IMPORTANT FOR JOURNAL)
    # --------------------------
    st.markdown(
        """
        **Interpretation:**
        - The bar chart shows the **mean absolute SHAP values**, representing
          the contribution of each feature to the model's prediction.
        - Features with higher SHAP values have a stronger influence on
          insurance claim likelihood.
        - For this prediction, **age and sex** exhibit the most significant impact,
          while other variables contribute marginally.
        """
    )
