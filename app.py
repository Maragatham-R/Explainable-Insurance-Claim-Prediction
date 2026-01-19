import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

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
        st.error(f"Claim  (Probability: {probability:.2f})")
    else:
        st.success(f"Not claim(Probability: {probability:.2f})")

    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input)

    # --- Handle classifier output ---
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_vals = np.array(shap_vals).flatten()

    # --- Feature names from model ---
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = [f"Feature {i}" for i in range(len(shap_vals))]

    # --- FORCE SAME LENGTH ---
    min_len = min(len(feature_names), len(shap_vals))
    shap_vals = shap_vals[:min_len]
    feature_names = feature_names[:min_len]

    # --- SORT BY IMPORTANCE ---
    order = np.argsort(np.abs(shap_vals))

    # --- REDUCE INDEX RANGE (TOP K FEATURES ONLY) ---
    TOP_K = min(10, len(order))   # <- THIS PREVENTS OUT OF RANGE
    order = order[-TOP_K:]

    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        np.array(feature_names)[order],
        shap_vals[order]
    )

    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Top Feature Contributions")

    st.pyplot(fig)
    st.markdown("""
    ### 🔍 How to Read This SHAP Chart

    - **Positive SHAP value (+)**  
      → The feature **increases the likelihood of an insurance claim**.  
      → It pushes the prediction **towards “Claim Likely”**.

    - **Negative SHAP value (–)**  
      → The feature **reduces the likelihood of an insurance claim**.  
      → It pushes the prediction **towards “Claim Not Likely”**.



    ### 📊 Feature Interpretation Example
    - Features with **longer bars** have **greater influence**
    - Bars to the **right (+)** increase risk
    - Bars to the **left (–)** decrease risk
    """)
    st.info(
        "Negative SHAP values reduce the insurer’s claim risk, while positive values increase it. "
        "A low probability indicates the customer is unlikely to file an insurance claim."
    )


