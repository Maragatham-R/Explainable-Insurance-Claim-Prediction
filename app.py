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
st.write("Predict insurance claims and understand feature impact using SHAP")

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
region_map = {"Southwest": 0, "Southeast": 1,_
