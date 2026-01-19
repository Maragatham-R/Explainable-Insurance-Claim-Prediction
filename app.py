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
    "This application predicts insurance claim likelihood and explains "
    "the prediction using SHAP-based feature contributions."
)

# --------------------------
# Load model
# --------------------------
model = joblib.load("insurance.pkl")

# --------------------------
# Sidebar – User Inputs
# --------------------------
st.sidebar.header("📋 Customer Details")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.number_input("Number of Children", 0, 10, 0)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox(
    "Region", ["Southwest", "Southeast", "Northwest", "Northeast"]
)
charges = st.sidebar.number_input("Medical Charges", 100.0, 100000.0, 5000.0)

predict_btn = st.sidebar.button("🔍 Predict Claim")

# --------------------------
# Main Panel – Prediction & Explanation
# --------------------------
if predict_btn:
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
    # SHAP Explanation → PIE CHART
    # --------------------------
    st.subheader("🔎 Explanation of Prediction")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input)

    # Binary classification → use positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    # Convert SHAP values to absolute contributions
    feature_names = user_input.columns
    shap_abs = np.abs(shap_vals)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": shap_abs
    })

    # Take top features only
    shap_df = shap_df.sort_values(
        by="Contribution", ascending=False
    ).head(5)

    # Normalize to percentages
    shap_df["Contribution (%)"] = (
        shap_df["Contribution"] / shap_df["Contribution"].sum()
    ) * 100

    # --------------------------
    # Pie Chart
    # --------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        shap_df["Contribution (%)"],
        labels=shap_df["Feature"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Feature Contribution to Prediction")

    st.pyplot(fig)

    # --------------------------
    # Simple Explanation Text
    # --------------------------
    st.markdown(
        """
        **How to interpret this chart:**
        - The pie chart shows the **relative contribution** of each feature
          to the model’s decision.
        - Larger slices indicate stronger influence on the insurance claim prediction.
        - This visualization is derived from **SHAP values**, ensuring faithful explainability.
        """
    )
