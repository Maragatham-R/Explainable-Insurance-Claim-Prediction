import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

st.title("Explainable Insurance Claim Prediction")
st.write("Predict insurance claims and understand feature impact using SHAP")

model = joblib.load("insurance.pkl")

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

sex_val = 0 if sex == "Female" else 1
smoker_val = 0 if smoker == "No" else 1
region_map = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}

user_input = pd.DataFrame(
    [[age, sex_val, bmi, children, smoker_val, region_map[region], charges]],
    columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"]
)

if st.sidebar.button("Predict Claim"):
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    if prediction == 1:
        st.error(f"Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"Claim Not Likely (Probability: {probability:.2f})")

    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input)

    # Handle binary vs single-output models
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    fig = plt.figure()
    shap.summary_plot(
        shap_vals,
        user_input,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)
