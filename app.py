# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import streamlit.components.v1 as components

# --------------------------
# Load model
# --------------------------
model = joblib.load("insurance.pkl")

st.set_page_config(page_title="Insurance Claim Prediction", layout="wide")
st.title("🛡️ Explainable Insurance Claim Prediction")

st.write("Predict insurance claim likelihood with SHAP-based explanations.")

# --------------------------
# User Inputs
# --------------------------
with st.form("input_form"):
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ["Female", "Male"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox(
        "Region", ["Southwest", "Southeast", "Northwest", "Northeast"]
    )
    charges = st.number_input("Charges", 100.0, 100000.0, 5000.0)

    submit = st.form_submit_button("🔍 Predict Claim")

# --------------------------
# Prediction + SHAP
# --------------------------
if submit:
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

    # Prediction
    pred = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ Claim Likely (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {prob:.2f})")

    # --------------------------
    # SHAP Explanation (FIXED)
    # --------------------------
    explainer = shap.Explainer(model)
    shap_values = explainer(user_input)

    st.subheader("🔎 SHAP Explanation")

    components.html(
        shap.force_plot(
            shap_values.base_values[0],
            shap_values.values[0],
            user_input.iloc[0],   # ✅ MUST be 1D
            feature_names=user_input.columns
        ).html(),
        height=350,
    )
