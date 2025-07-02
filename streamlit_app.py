# streamlit_app.py

import streamlit as st
import pandas as as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="🩺 Stroke Risk Predictor", layout="centered")

# --- Load Model ---
MODEL_PATH = 'stroke_risk_model_revised.pkl'

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure 'stroke_risk_model_revised.pkl' is uploaded.")
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    model = saved['model']
    threshold = saved.get('threshold', 0.01)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# --- Streamlit UI ---
st.title("🩺 Stroke Risk Prediction Tool")
st.markdown("Enter patient health metrics below to assess stroke risk.")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 35)

ever_married = st.radio("Ever Married?", ["Yes", "No"])

work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"]
)

residence_type = st.radio("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox(
    "Smoking Status",
    ["never_smoked", "formerly_smoked", "smokes", "Unknown"]
)

hypertension = st.checkbox("Hypertension")
heart_disease = st.checkbox("Heart Disease")

avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

# Manually encode categorical variables
gender_Male = 1 if gender == "Male" else 0
ever_married_Yes = 1 if ever_married == "Yes" else 0

work_type_Private = 1 if work_type == "Private" else 0
work_type_Self_employed = 1 if work_type == "Self-employed" else 0
work_type_Govt_job = 1 if work_type == "Govt_job" else 0
work_type_children = 1 if work_type == "Children" else 0
work_type_Never_worked = 1 if work_type == "Never_worked" else 0

Residence_type_Urban = 1 if residence_type == "Urban" else 0

smoking_status_formerly_smoked = 1 if smoking_status == "formerly_smoked" else 0
smoking_status_never_smoked = 1 if smoking_status == "never_smoked" else 0
smoking_status_smokes = 1 if smoking_status == "smokes" else 0

# Prepare DataFrame with correct structure
input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [1 if hypertension else 0],
    'heart_disease': [1 if heart_disease else 0],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'gender_Male': [gender_Male],
    'ever_married_Yes': [ever_married_Yes],
    'work_type_Private': [work_type_Private],
    'work_type_Self_employed': [work_type_Self_employed],
    'work_type_Govt_job': [work_type_Govt_job],
    'work_type_children': [work_type_children],
    'work_type_Never_worked': [work_type_Never_worked],
    'Residence_type_Urban': [Residence_type_Urban],
    'smoking_status_formerly smoked': [smoking_status_formerly_smoked],
    'smoking_status_never smoked': [smoking_status_never_smoked],
    'smoking_status_smokes': [smoking_status_smokes]
})

# Make prediction
proba = model.predict_proba(input_data)[0][1]
risk_level = "High Risk" if proba >= threshold else "Low Risk"

# Display result
st.markdown("### ⚠️ Stroke Risk Level: **{}**".format(risk_level))
st.progress(proba)
st.write("Probability of Stroke: {:.2f}%".format(proba * 100))
