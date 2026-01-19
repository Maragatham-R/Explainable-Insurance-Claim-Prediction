import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

st.title("Explainable Insurance Claim Prediction")
st.write("Predict insurance claims with transparent SHAP explanations.")

# ---------------------------
# Load Pre-trained Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("insurance.pkl")

model = load_model()

# ---------------------------
# Initialize SHAP Explainer
# (NO X_train needed)
# ---------------------------
explainer = shap.TreeExplainer(model)

# ---------------------------
# Sidebar - User Input
# ---------------------------
st.sidebar.header("📑Enter Customer Details")

age = st.sidebar.slider("Age", 18, 65, 30)

gender_input = st.sidebar.selectbox("Gender", ["Female", "Male"])
sex = 0 if gender_input == "Female" else 1

bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)

children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4])

smoker_input = st.sidebar.selectbox("Smoker", ["No", "Yes"])
smoker = 0 if smoker_input == "No" else 1

region_input = st.sidebar.selectbox(
    "Region",
    ["Southwest", "Southeast", "Northwest", "Northeast"]
)

region_map = {
    "Southwest": 0,
    "Southeast": 1,
    "Northwest": 2,
    "Northeast": 3
}
region = region_map[region_input]

charges = st.sidebar.number_input(
    "Medical Charges",
    min_value=0.0,
    value=5000.0
)

# ---------------------------
# Create Input DataFrame
# ---------------------------
user_input = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges
}])

# ---------------------------
# Prediction & SHAP Explanation
# ---------------------------
if st.button("🔍 Predict Claim"):

    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Claim Not Likely (Probability: {probability:.2f})")

    # ---------------------------
    # SHAP Explanation
    # ---------------------------
    st.subheader("📊 Why this prediction was made")

    # Compute SHAP values
shap_values = explainer.shap_values(user_input)

# Check if shap_values is a list (binary classification)
if isinstance(shap_values, list):
    # Class 1 = Claim
    shap_values_claim = shap_values[1]
    expected_value = explainer.expected_value[1]
else:
    # shap_values is a single array
    shap_values_claim = shap_values
    expected_value = explainer.expected_value

# Create SHAP force plot
fig = shap.force_plot(
    expected_value,
    shap_values_claim,
    user_input,
    matplotlib=True
)

# Display in Streamlit
st.pyplot(fig, bbox_inches='tight')

st.pyplot(fig, bbox_inches="tight")

    # ---------------------------
    # Simple Explanation Text
    # ---------------------------
    st.subheader("🧠 Explanation in Simple Terms")

    shap_df = pd.DataFrame({
        "Feature": user_input.columns,
        "Impact": shap_values[1][0]
    }).sort_values(by="Impact", ascending=False)

    for _, row in shap_df.iterrows():
        if row["Impact"] > 0:
            st.write(f"🔺 **{row['Feature']}** increased the chance of a claim")
        else:
            st.write(f"🔻 **{row['Feature']}** reduced the chance of a claim")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Explainable AI Deployment using SHAP & Streamlit")

