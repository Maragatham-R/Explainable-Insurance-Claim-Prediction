import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

st.title("🛡️ Explainable Insurance Claim Prediction")
st.write("Prediction with user-friendly SHAP explanation")

# -----------------------------
# LOAD PIPELINE MODEL
# -----------------------------
model = joblib.load("insurance.pkl")

# -----------------------------
# SESSION STATE
# -----------------------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# -----------------------------
# SIDEBAR INPUTS (RAW FEATURES)
# -----------------------------
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

# -----------------------------
# RAW INPUT DATAFRAME
# -----------------------------
raw_input = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges
}])

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.sidebar.button("🔍 Predict Claim"):
    st.session_state.prediction_done = True
    st.session_state.raw_input = raw_input

    pred = model.predict(raw_input)[0]
    prob = model.predict_proba(raw_input)[0][1]

    st.session_state.pred = pred
    st.session_state.prob = prob

# -----------------------------
# SHOW RESULT
# -----------------------------
if st.session_state.prediction_done:

    if st.session_state.pred == 1:
        st.error(f"⚠️ Claim Likely (Probability: {st.session_state.prob:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {st.session_state.prob:.2f})")

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    if st.button("📊 Show SHAP Explanation"):

        st.subheader("🔎 SHAP Explanation (Top Factors)")

        # SHAP explainer (PIPELINE SAFE)
        explainer = shap.Explainer(model)
        shap_values = explainer(st.session_state.raw_input)

        # Extract values safely
        values = np.abs(shap_values.values[0])
        features = shap_values.feature_names

        shap_df = pd.DataFrame({
            "Feature": features,
            "Impact": values
        }).sort_values(by="Impact", ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.set_xlabel("Impact on Prediction")
        ax.set_title("Top Contributing Features")

        st.pyplot(fig)

        st.info(
            "This chart shows how each feature influenced the prediction. "
            "Higher bars indicate stronger impact."
        )
