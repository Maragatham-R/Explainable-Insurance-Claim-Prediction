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

# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load("insurance.pkl")

# --------------------------
# GET FEATURE NAMES FROM MODEL (CRITICAL FIX)
# --------------------------
model_features = model.feature_names_in_

# --------------------------
# SESSION STATE
# --------------------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False

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

# IMPORTANT: follow model feature order
user_input = pd.DataFrame(
    [[age, sex_val, bmi, children, smoker_val, region_map[region], charges]],
    columns=model_features
)

# --------------------------
# PREDICT BUTTON
# --------------------------
if st.sidebar.button("🔍 Predict Claim"):
    st.session_state.predicted = True
    st.session_state.user_input = user_input
    st.session_state.prediction = model.predict(user_input)[0]
    st.session_state.probability = model.predict_proba(user_input)[0][1]

# --------------------------
# SHOW RESULT
# --------------------------
if st.session_state.predicted:

    if st.session_state.prediction == 1:
        st.error(
            f"⚠️ Claim Likely (Probability: {st.session_state.probability:.2f})"
        )
    else:
        st.success(
            f"✅ Claim Not Likely (Probability: {st.session_state.probability:.2f})"
        )

    # --------------------------
    # SHAP BUTTON
    # --------------------------
    if st.button("📊 Show SHAP Explanation"):

        st.subheader("🔎 Model Explanation (SHAP Bar Chart)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(st.session_state.user_input)

        # Binary classification handling
        if isinstance(shap_values, list):
            shap_vals = (
                shap_values[1][0]
                if st.session_state.prediction == 1
                else shap_values[0][0]
            )
        else:
            shap_vals = shap_values[0]

        # Force 1D
        shap_vals = np.array(shap_vals).reshape(-1)

        # FINAL SAFE DATAFRAME
        shap_df = pd.DataFrame({
            "Feature": model_features,
            "Impact": np.abs(shap_vals)
        }).sort_values(by="Impact", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.set_xlabel("Feature Impact")
        ax.set_title("SHAP Feature Contribution")

        st.pyplot(fig)

        st.info(
            "SHAP bar chart visualizes the magnitude of each feature's "
            "contribution to the individual prediction."
        )
