import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Clinical Cluster Predictor", layout="centered")

st.title("🩺 Clinical Cohort & Cluster Predictor")
st.write("This application predicts clinical profiles into one of three distinct biological cohorts using key physical and metabolic indicators based on the Breast Cancer Coimbra Dataset.")

# Load pre-trained model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

# Input fields matching the original datar2 clinical columns
st.header("Patient Clinical Indicators")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=10, max_value=300, value=90)
    insulin = st.number_input("Insulin (µU/mL)", min_value=1.0, max_value=100.0, value=5.0, step=0.1)
    homa = st.number_input("HOMA-IR", min_value=0.1, max_value=50.0, value=1.0, step=0.1)

with col2:
    leptin = st.number_input("Leptin (ng/mL)", min_value=1.0, max_value=150.0, value=15.0, step=0.1)
    adiponectin = st.number_input("Adiponectin (µg/mL)", min_value=1.0, max_value=100.0, value=10.0, step=0.1)
    resistin = st.number_input("Resistin (ng/mL)", min_value=1.0, max_value=100.0, value=8.0, step=0.1)
    mcp_1 = st.number_input("MCP-1 (pg/mL)", min_value=10.0, max_value=2000.0, value=250.0, step=10.0)

# Predict
if st.button("Classify Profile", type="primary"):
    # Map inputs to feature list matching scaled training dataset columns
    input_data = pd.DataFrame([[
        age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1
    ]], columns=['age', 'bmi', 'glucose', 'insulin', 'homa', 'leptin', 'adiponectin', 'resistin', 'mcp_1'])

    # Scale inputs
    input_scaled = scaler.transform(input_data)

    # Classify
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Results")

    if prediction == 0:
        st.success("🎯 **Predicted Cluster: 0 (Healthy Profile)**")
        st.write("**Characteristics**: Typically represents normal glycemic metrics, lower inflammatory marker load, and healthy-leaning clinical baseline values.")
    elif prediction == 1:
        st.error("⚠️ **Predicted Cluster: 1 (High Metabolic/Glycemic Risk Profile)**")
        st.write("**Characteristics**: Represents extremely high blood glucose levels, insulin, and high HOMA-IR levels requiring attention.")
    elif prediction == 2:
        st.warning("⚡ **Predicted Cluster: 2 (Increased Obesity & Leptin Profile)**")
        st.write("**Characteristics**: Marked by elevated BMIs and significantly high Leptin levels, signaling increased systemic obesity indices.")
