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
model_features = model.feature_names_in_

# --------------------------
# SESSION STATE INIT
# --------------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "probability" not in st.session_state:
    st.session_state.probability = None

# --------------------------
# SIDEBAR INPUTS
# --------------------------
st.sidebar.header("User Inputs")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Gender", ["female", "male"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.number_input("Children", 0, 10, 0)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox(
    "Region", ["southwest", "southeast", "northwest", "northeast"]
)
charges = st.sidebar.number_input("Charges", 100.0, 100000.0, 5000.0)

# --------------------------
# BUILD FEATURE VECTOR (MATCH MODEL)
# --------------------------
user_input = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)

# Numeric features
for col, val in {"age": age, "bmi": bmi, "children": children, "charges": charges}.items():
    if col in user_input.columns:
        user_input[col] = val

# Categorical features (only if exist)
if f"sex_{sex}" in user_input.columns:
    user_input[f"sex_{sex}"] = 1
if f"smoker_{smoker}" in user_input.columns:
    user_input[f"smoker_{smoker}"] = 1
if f"region_{region}" in user_input.columns:
    user_input[f"region_{region}"] = 1

# --------------------------
# PREDICTION BUTTON
# --------------------------
if st.sidebar.button("🔍 Predict Claim"):
    st.session_state.user_input = user_input
    st.session_state.prediction = model.predict(user_input)[0]
    st.session_state.probability = model.predict_proba(user_input)[0][1]

# --------------------------
# SHOW PREDICTION
# --------------------------
if st.session_state.user_input is not None:
    if st.session_state.prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {st.session_state.probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {st.session_state.probability:.2f})")

    # --------------------------
    # SHOW SHAP BUTTON
    # --------------------------
    if st.button("📊 Show SHAP Explanation"):

        st.subheader("🔎 SHAP Feature Contribution")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(st.session_state.user_input)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] if st.session_state.prediction == 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        shap_vals = np.abs(shap_vals).flatten()

        # Build DataFrame
        shap_df = pd.DataFrame({"Feature": model_features, "Impact": shap_vals})
        shap_df = shap_df.sort_values(by="Impact", ascending=True).tail(10)

        # Bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Top Feature Contributions")

        st.pyplot(fig)

        st.info(
            "The bar chart shows the magnitude of each feature's contribution "
            "to the prediction."
        )
