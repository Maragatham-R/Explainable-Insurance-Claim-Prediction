import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

st.title("🛡️ Explainable Insurance Claim Prediction")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("insurance.pkl")

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
# ENCODE CATEGORICALS
# -----------------------------
sex_val = 0 if sex == "Female" else 1
smoker_val = 0 if smoker == "No" else 1
region_map = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}

# -----------------------------
# INPUT DATAFRAME
# -----------------------------
user_input = pd.DataFrame(
    [[age, sex_val, bmi, children, smoker_val, region_map[region], charges]],
    columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"]
)

# -----------------------------
# PREDICT
# -----------------------------
if st.sidebar.button("🔍 Predict Claim"):
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.subheader("🔎 SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(user_input)

    # Waterfall plot (BEST for single prediction)
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
