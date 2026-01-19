# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Insurance Claim Prediction",
    layout="wide"
)

st.title("🛡️ Explainable Insurance Claim Prediction")
st.write("Predict insurance claim likelihood with SHAP-based explanations.")

# --------------------------
# Load trained model
# --------------------------
model = joblib.load("insurance.pkl")

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

    # Prediction
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")

    # --------------------------
    # SHAP Explanation (STABLE)
    # --------------------------
    st.subheader("🔎 Explanation of Prediction")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_input)

fig, ax = plt.subplots()
shap.summary_plot(
    shap_values[1],       # class 1 = claim
    user_input,
    plot_type="bar",
    show=False
)
st.pyplot(fig)
