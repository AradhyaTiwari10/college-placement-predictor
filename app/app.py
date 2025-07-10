import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("../models/placement_logistic_model.pkl")
scaler = joblib.load("../models/scaler.pkl")


st.title("🎓 College Placement Predictor with Smart Suggestions")

st.markdown("""
This tool predicts the placement chance of a student based on academic and personal data.

> ⚠️ **Note:** This prediction is a simulation using synthetic data. It is for learning purposes only and doesn't guarantee real-life results.
""")

st.sidebar.header("🔧 Enter Student Details")


iq = st.sidebar.number_input("IQ Score (Typical: 80–130)", 50, 150, 110)
prev_sem_result = st.sidebar.number_input("Previous Semester GPA (0 - 10)", 0.0, 10.0, 7.0)
cgpa = st.sidebar.number_input("Cumulative GPA (CGPA) (0 - 10)", 0.0, 10.0, 7.0)
academic_performance = st.sidebar.slider("Academic Performance (1-10)", 1, 10, 7)
internship_experience = st.sidebar.slider("Number of Internships", 0, 5, 1)
extra_curricular_score = st.sidebar.slider("Extra-Curricular Score (0-10)", 0, 10, 5)
communication_skills = st.sidebar.slider("Communication Skills (1-10)", 1, 10, 7)
projects_completed = st.sidebar.slider("Projects Completed", 0, 5, 2)


if st.button("🔍 Predict Placement"):
    input_data = pd.DataFrame([[
        iq, prev_sem_result, cgpa, academic_performance,
        internship_experience * 10,  
        extra_curricular_score, communication_skills, projects_completed
    ]], columns=[
        'IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
        'Internship_Experience', 'Extra_Curricular_Score',
        'Communication_Skills', 'Projects_Completed'
    ])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)[0][1]


    if prediction[0] == 1:
        st.success("✅ Prediction Result: Placed")
    else:
        st.error("❌ Prediction Result: Not Placed")

    st.write(f"Prediction Confidence: {prediction_proba:.2f}")


    st.subheader("📊 Personalized Suggestions Based on Placed Students")
    dataset = pd.read_csv("../data/college_student_placement_dataset.csv")
    dataset['Internship_Experience'] = dataset['Internship_Experience'].map({'Yes': 1, 'No': 0})
    dataset['Placement'] = dataset['Placement'].map({'Yes': 1, 'No': 0})
    dataset['Internship_Experience'] = dataset['Internship_Experience'] * 10  

    placed_data = dataset[dataset['Placement'] == 1]
    avg_scores = placed_data.mean()

    suggestions = []
    feature_inputs = {
        'IQ': iq,
        'Prev_Sem_Result': prev_sem_result,
        'CGPA': cgpa,
        'Academic_Performance': academic_performance,
        'Internship_Experience': internship_experience * 10,
        'Extra_Curricular_Score': extra_curricular_score,
        'Communication_Skills': communication_skills,
        'Projects_Completed': projects_completed
    }

    for feature, user_val in feature_inputs.items():
        avg_val = avg_scores[feature]
        if user_val < avg_val:
            suggestions.append(f"📈 Improve **{feature.replace('_', ' ')}** (Average among placed students: {avg_val:.2f})")

    if suggestions:
        for s in suggestions:
            st.markdown(s)
    else:
        st.info("✅ Your profile already matches typical placed student averages!")

