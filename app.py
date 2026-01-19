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

# --------------------------
# Title
# --------------------------
st.title("🛡️ Explainable Insurance Claim Prediction")
st.write(
    "Predict the likelihood of an insurance claim and understand the decision "
    "using SHAP-based explainability."
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
# Main Panel – Prediction & SHAP
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
    # SHAP Explanation (FIXED & VISIBLE)
    # --------------------------
    st.subheader("🔎 Model Explanation (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input)

    # For binary classification → focus on positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # IMPORTANT: Clear previous plots
    plt.clf()

    # SHAP bar plot (journal-safe)
    shap.summary_plot(
        shap_vals,
        user_input,
        plot_type="bar",
        max_display=7,
        show=False
    )

    # Display the SHAP figure
    st.pyplot(plt.gcf())

    # --------------------------
    # Plain-English Explanation
    # --------------------------
    st.markdown(
        """
        **How to read this chart:**
        - The bar chart represents **mean absolute SHAP values**
        - Higher values indicate stronger influence on the prediction
        - For this case, **age and sex** contribute most to the claim decision
        """
    )
