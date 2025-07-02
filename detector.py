import streamlit as st
import pandas as pd
import joblib

# --- Load imblearn pipeline dependency for deserialization ---
from imblearn.pipeline import Pipeline as imbpipeline  # Required if model uses imbalanced-learn pipeline

# --- Load model and threshold ---
MODEL_PATH = 'stroke_risk_model_1.pkl'

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved['pipeline'], saved['threshold']

pipeline, best_threshold = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Stroke Risk Predictor")
st.write("This tool estimates the risk of stroke based on basic health indicators.")

# --- Input Form ---
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.radio("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", ["never_smoked", "formerly_smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    # Format input data as DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }])

    # Predict probability
    try:
        probability = pipeline.predict_proba(input_data)[0][1]
        risk_level = "High Risk" if probability >= best_threshold else "Low Risk"

        # Display Results
        st.markdown("### ✅ Prediction Result")
        st.success(f"*Stroke Risk Level:* {risk_level}")
        st.info(f"*Probability of Stroke:* {probability * 100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
