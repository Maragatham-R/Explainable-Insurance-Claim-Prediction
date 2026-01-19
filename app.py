import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

st.title("🛡️ Explainable Insurance Claim Prediction")
st.write("Predict insurance claims and see feature contributions using SHAP")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("insurance.pkl")  # model trained on numeric features

# -----------------------------
# SESSION STATE
# -----------------------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
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

# -----------------------------
# ENCODE CATEGORICALS (NUMERIC MODEL)
# -----------------------------
sex_val = 0 if sex == "Female" else 1
smoker_val = 0 if smoker == "No" else 1
region_map = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
region_val = region_map[region]

# -----------------------------
# CREATE USER INPUT DATAFRAME
# -----------------------------
user_input = pd.DataFrame(
    [[age, sex_val, bmi, children, smoker_val, region_val, charges]],
    columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"],
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.sidebar.button("🔍 Predict Claim"):
    st.session_state.predicted = True
    st.session_state.user_input = user_input
    st.session_state.prediction = model.predict(user_input)[0]
    st.session_state.probability = model.predict_proba(user_input)[0][1]

# -----------------------------
# SHOW PREDICTION
# -----------------------------
if st.session_state.predicted:
    if st.session_state.prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {st.session_state.probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {st.session_state.probability:.2f})")

    # -----------------------------
    # SHAP BUTTON
    # -----------------------------
    if st.button("📊 Show SHAP Explanation"):

        st.subheader("🔎 SHAP Feature Contributions")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(st.session_state.user_input)

        # Binary classification handling
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] if st.session_state.prediction == 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        # Flatten to 1D to avoid ValueError
        shap_vals = np.abs(np.array(shap_vals).flatten())

        # Build DataFrame for bar chart
        shap_df = pd.DataFrame({
            "Feature": user_input.columns,
            "Impact": shap_vals
        }).sort_values(by="Impact", ascending=True)

        # Plot top features
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.set_xlabel("Impact on Prediction")
        ax.set_title("Feature Contribution (SHAP)")

        st.pyplot(fig)

        st.info(
            "This bar chart shows the contribution of each feature to the prediction. "
            "Higher bars indicate stronger influence."
        )
