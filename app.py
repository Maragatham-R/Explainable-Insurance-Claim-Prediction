import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

st.title("🛡️ Explainable Insurance Claim Prediction")
st.write("Prediction with SHAP-based explainability")

# ------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------
model = joblib.load("insurance.pkl")

# IMPORTANT: training feature space
model_features = model.feature_names_in_

# ------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------
st.sidebar.header("User Input Parameters")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Gender", ["female", "male"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.number_input("Children", 0, 10, 0)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox(
    "Region", ["southwest", "southeast", "northwest", "northeast"]
)
charges = st.sidebar.number_input("Charges", 100.0, 100000.0, 5000.0)

# ------------------------------------------------
# CREATE FULL FEATURE VECTOR (NO MISMATCH)
# ------------------------------------------------
user_input = pd.DataFrame(
    np.zeros((1, len(model_features))),
    columns=model_features
)

# Assign numeric features
user_input["age"] = age
user_input["bmi"] = bmi
user_input["children"] = children
user_input["charges"] = charges

# Assign encoded categorical features
user_input[f"sex_{sex}"] = 1
user_input[f"smoker_{smoker}"] = 1
user_input[f"region_{region}"] = 1

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
if st.sidebar.button("🔍 Predict Claim"):

    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")

    # ------------------------------------------------
    # SHAP EXPLANATION BUTTON
    # ------------------------------------------------
    if st.button("📊 Show SHAP Explanation"):

        st.subheader("🔎 SHAP Feature Importance")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_input)

        # -------- SHAP SHAPE FIX (CRITICAL) --------
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # positive class
        else:
            shap_vals = shap_values[0]

        shap_vals = np.abs(shap_vals).flatten()

        # GUARANTEED SAME LENGTH
        shap_df = pd.DataFrame({
            "Feature": model_features,
            "Impact": shap_vals
        })

        shap_df = shap_df.sort_values(
            by="Impact", ascending=True
        ).tail(10)

        # ------------------------------------------------
        # BAR CHART
        # ------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Top Contributing Features")

        st.pyplot(fig)

        st.info(
            "The bar chart shows how strongly each feature "
            "contributed to the prediction."
        )
