import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="SHAP Explainable ML App", layout="wide")

st.title("Explainable AI using SHAP")
st.write("User inputs on sidebar and SHAP explanation in the center")

# --------------------------
# SAMPLE DATA (replace with your dataset)
# --------------------------
# Example dataset (binary classification)
data = pd.DataFrame({
    "age": [25, 45, 35, 50, 23, 40],
    "sex": [0, 1, 1, 0, 0, 1],
    "bmi": [22.5, 30.1, 27.3, 26.8, 21.9, 28.4],
    "bp": [120, 140, 130, 135, 118, 145],
    "target": [0, 1, 1, 1, 0, 1]
})

X = data.drop("target", axis=1)
y = data["target"]

# --------------------------
# TRAIN MODEL
# --------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --------------------------
# SIDEBAR INPUTS
# --------------------------
st.sidebar.header("User Input Parameters")

age = st.sidebar.slider("Age", 18, 80, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
bp = st.sidebar.slider("Blood Pressure", 90, 180, 120)

sex_val = 1 if sex == "Male" else 0

user_input = pd.DataFrame({
    "age": [age],
    "sex": [sex_val],
    "bmi": [bmi],
    "bp": [bp]
})

# --------------------------
# PREDICTION
# --------------------------
prediction = model.predict(user_input)[0]
prediction_proba = model.predict_proba(user_input)[0][1]

st.subheader("Prediction Result")
st.write(f"**Predicted Class:** {prediction}")
st.write(f"**Prediction Probability:** {prediction_proba:.2f}")

# --------------------------
# SHAP EXPLANATION (FIXED)
# --------------------------
st.subheader("SHAP Feature Contribution (Pie Chart)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_input)

# ---- FIX: force SHAP values to 1D ----
if isinstance(shap_values, list):
    shap_vals = shap_values[1][0].flatten()
else:
    shap_vals = shap_values[0].flatten()

feature_names = user_input.columns
shap_abs = np.abs(shap_vals)

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": shap_abs
})

# Take top features
shap_df = shap_df.sort_values(
    by="Contribution", ascending=False
)

# Convert to percentage
shap_df["Contribution (%)"] = (
    shap_df["Contribution"] / shap_df["Contribution"].sum()
) * 100

# --------------------------
# PIE CHART
# --------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    shap_df["Contribution (%)"],
    labels=shap_df["Feature"],
    autopct="%1.1f%%",
    startangle=90
)
ax.set_title("Feature Contribution to Model Prediction")

st.pyplot(fig)

# --------------------------
# JOURNAL NOTE
# --------------------------
st.info(
    "SHAP values are normalized and visualized as a pie chart to enhance "
    "interpretability of individual predictions."
)
