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
# LOAD MODEL & BACKGROUND DATA
# --------------------------
model = joblib.load("insurance.pkl")

# IMPORTANT: background data (from training)
# Load the SAME training data used for model
background = pd.read_csv("insurance.csv")
X_background = background.drop("target", axis=1)

# --------------------------
# SIDEBAR INPUTS
# --------------------------
st.sidebar.header("User Inputs")

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
# PREDICTION
# --------------------------
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]

if prediction == 1:
    st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
else:
    st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")

# --------------------------
# SHAP BAR PLOT (CORRECT WAY)
# --------------------------
st.subheader("🔎 Model Explanation (SHAP Bar Plot)")

# Create explainer with background data
explainer = shap.TreeExplainer(model, X_background)

# Compute SHAP values
shap_values = explainer.shap_values(user_input)

# Select correct class
if isinstance(shap_values, list):
    shap_vals = shap_values[1][0] if prediction == 1 else shap_values[0][0]
else:
    shap_vals = shap_values[0]

# Convert to DataFrame (FIXES EMPTY PLOT)
shap_df = pd.DataFrame({
    "Feature": user_input.columns,
    "SHAP Value": np.abs(shap_vals)
}).sort_values(by="SHAP Value", ascending=True)

# --------------------------
# MANUAL BAR PLOT (MOST STABLE)
# --------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
ax.set_xlabel("Impact on Model Output")
ax.set_title("Feature Importance for Current Prediction")

st.pyplot(fig)
