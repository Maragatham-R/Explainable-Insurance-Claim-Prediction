# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import streamlit.components.v1 as components

# --------------------------
# Load pre-trained model
# --------------------------
model = joblib.load("rf_model.pkl")  # Make sure this file is in the same folder

# --------------------------
# Streamlit App UI
# --------------------------
st.set_page_config(page_title="Insurance Claim Prediction", layout="wide")
st.title("🛡️ Explainable Insurance Claim Prediction App")

st.write("Enter customer details to predict whether an insurance claim is likely, with feature explanations using SHAP.")

# --------------------------
# User Input Form
# --------------------------
with st.form(key="claim_form"):
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ["Female", "Male"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])
    charges = st.number_input("Charges", min_value=100.0, max_value=100000.0, value=5000.0)
    
    submit_button = st.form_submit_button(label="🔍 Predict Claim")

# --------------------------
# Preprocess User Input
# --------------------------
if submit_button:
    # Convert categorical inputs to numeric as per dataset encoding
    sex_val = 0 if sex == "Female" else 1
    smoker_val = 0 if smoker == "No" else 1
    region_map = {"Southwest":0, "Southeast":1, "Northwest":2, "Northeast":3}
    region_val = region_map[region]
    
    # Create single-row DataFrame
    user_input = pd.DataFrame([[age, sex_val, bmi, children, smoker_val, region_val, charges]],
                              columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"])
    
    # --------------------------
    # Prediction
    # --------------------------
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]  # Probability of claim (class 1)
    
    if prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")
    
    # --------------------------
    # SHAP Explanation (uses model internal tree for background)
    # --------------------------
    explainer = shap.TreeExplainer(model)
    user_input_array = user_input.values
    shap_values = explainer(user_input_array)

    st.subheader("🔎 SHAP Feature Explanation")
    st.write("This plot shows how each feature contributed to the prediction.")

    # Display interactive SHAP force plot
    components.html(
        shap.force_plot(
            shap_values.base_values[0],
            shap_values.values[0],
            user_input_array
        ).html(),
        height=350
    )
