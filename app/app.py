import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("../models/placement_logistic_model.pkl")
scaler = joblib.load("../models/scaler.pkl")


st.title("🎓 College Placement Predictor")
st.write("Predict whether a student will get placed based on academic and professional profile.")


st.subheader("Enter Student Details:")

iq = st.number_input("IQ Score", min_value=50, max_value=150, value=110)
prev_sem_result = st.number_input("Previous Semester GPA", min_value=0.0, max_value=10.0, value=7.0)
cgpa = st.number_input("Cumulative GPA (CGPA)", min_value=0.0, max_value=10.0, value=7.5)
academic_performance = st.slider("Academic Performance (1-10)", min_value=1, max_value=10, value=7)
internship_experience = st.number_input("Internship Experience (Number of Internships)", min_value=0, max_value=10, value=1)
extra_curricular_score = st.slider("Extra-Curricular Score (0-10)", min_value=0, max_value=10, value=5)
communication_skills = st.slider("Communication Skills (1-10)", min_value=1, max_value=10, value=7)
projects_completed = st.number_input("Projects Completed (Major Projects)", min_value=0, max_value=10, value=2)


if st.button("Predict Placement"):
    # Prepare DataFrame
    input_data = pd.DataFrame([[
        iq, prev_sem_result, cgpa, academic_performance,
        internship_experience, extra_curricular_score,
        communication_skills, projects_completed
    ]], columns=[
        'IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
        'Internship_Experience', 'Extra_Curricular_Score',
        'Communication_Skills', 'Projects_Completed'
    ])


    input_data['Internship_Experience'] = input_data['Internship_Experience'] * 10


    input_data_scaled = scaler.transform(input_data)


    proba = model.predict_proba(input_data_scaled)[0][1]
    st.write(f"🔢 Predicted Placement Probability: **{proba:.2f}**")

    threshold = 0.4  
    placement = 1 if proba >= threshold else 0

    result = "🎉 Placed" if placement == 1 else "❌ Not Placed"
    st.subheader("Prediction Result:")
    st.success(result)


    st.info("""
    ⚠️ **Disclaimer:**  
    This prediction is based on machine learning using synthetic data for educational purposes only.  
    It does not guarantee real-life placement results. Actual placement depends on many factors like interviews, market conditions, and more.
    """)


