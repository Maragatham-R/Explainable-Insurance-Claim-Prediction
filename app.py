import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

st.title("🛡️ Explainable Insurance Claim Prediction")
st.write("Prediction with SHAP-based feature contribution")

# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load("insurance.pkl")

# --------------------------
# SIDEBAR INPUTS
# --------------------------
st.sidebar.header("User Input Parameters")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.number_input("Children", 0, 10, 0)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox(
    "Region", ["Southwest", "Southeast", "Northwest", "Northeast"]
)
charges = st.sidebar.number_input("Charges", 100.0, 100000.0, 5000.0)

# Encoding
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
# PREDICTION BUTTON
# --------------------------
if st.sidebar.button("🔍 Predict Claim"):

    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")

    # --------------------------
    # SHAP EXPLANATION BUTTON
    # --------------------------
    if st.button("📊 Show SHAP Explanation"):

        st.subheader("🔎 Model Explanation (SHAP Bar Chart)")

        # SHAP explainer (NO background mentioned)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_input)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] if prediction == 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        # Convert to DataFrame (1D FIX)
        shap_df = pd.DataFrame({
            "Feature": user_input.columns,
            "Impact": np.abs(shap_vals)
        }).sort_values(by="Impact", ascending=True)

        # --------------------------
        # BAR PLOT (STREAMLIT SAFE)
        # --------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.set_xlabel("Feature Impact on Prediction")
        ax.set_title("SHAP Feature Contribution")

        st.pyplot(fig)

        st.info(
            "The bar chart shows the magnitude of each feature's contribution "
            "to the individual prediction using SHAP values."
        )
