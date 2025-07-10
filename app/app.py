import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Load Model and Scaler
model = joblib.load("../models/placement_logistic_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# ✅ Load Dataset for Analysis
df = pd.read_csv("../data/college_student_placement_dataset.csv")

# ✅ Preprocessing
df['Internship_Experience'] = df['Internship_Experience'].map({'Yes': 1, 'No': 0})
df['Placement'] = df['Placement'].map({'Yes': 1, 'No': 0})
df['Internship_Experience'] = df['Internship_Experience'] * 10  # Amplified in training

# ✅ Feature Columns
features = [
    'IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
    'Internship_Experience', 'Extra_Curricular_Score',
    'Communication_Skills', 'Projects_Completed'
]

# ✅ UI
st.title("📊 College Placement Predictor")
st.write("⚠️ Note: This is a prediction model for learning purposes. Actual placement depends on many factors.")

st.header("🎓 Enter Your Details")


user_data = {}
for feature in features:
    if feature == 'Internship_Experience':
        user_data[feature] = st.number_input("Number of Internships Completed", min_value=0, step=1)
    elif feature in ['Academic_Performance', 'Extra_Curricular_Score', 'Communication_Skills']:
        user_data[feature] = st.slider(f"{feature.replace('_', ' ')} (0-10)", 0, 10)
    else:
        user_data[feature] = st.number_input(f"{feature.replace('_', ' ')}", step=0.1)

# ✅ Prediction
if st.button("Predict Placement"):
    input_df = pd.DataFrame([user_data])
    input_df['Internship_Experience'] *= 10  
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.success("🎉 Congratulations! The model predicts you are likely to be placed.")
    else:
        st.error("❌ Unfortunately, the model predicts you may not get placed.")
        st.info("🔍 Here’s what you can improve based on placed students' averages:")

        # ✅ Suggest Improvements
        placed_students = df[df['Placement'] == 1]
        avg_scores = placed_students[features].mean()

        suggestions = []
    for col in features:
        your_score = user_data[col]
        avg_score = avg_scores[col] / (10 if col == 'Internship_Experience' else 1)  # Adjust internship scaling
        
        # Round appropriately
        if col in ['Internship_Experience', 'Projects_Completed']:
            your_score_display = int(your_score)
            avg_score_display = int(round(avg_score))
        else:
            your_score_display = round(your_score, 2)
            avg_score_display = round(avg_score, 2)
        
        if your_score < avg_score:
            suggestion = (
                f"- **{col.replace('_', ' ')}**: Your score ({your_score_display}) "
                f"is below the average of placed students ({avg_score_display})."
            )
            suggestions.append(suggestion)

        
        if suggestions:
            st.markdown("\n".join(suggestions))
        else:
            st.info("✅ You already match or exceed average scores of placed students! Focus on applying actively.")

