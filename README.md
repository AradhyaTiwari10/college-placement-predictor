# College Placement Predictor

A Streamlit web app that predicts the likelihood of a college student getting placed based on academic and extracurricular features. It uses a machine learning model trained on real placement data and provides actionable suggestions for improvement.

## Features
- Predicts placement chances based on:
  - IQ Score
  - Previous Semester GPA
  - Cumulative GPA (CGPA)
  - Academic Performance (1-10)
  - Internship Experience (number of internships)
  - Extra-Curricular Score (0-10)
  - Communication Skills (1-10)
  - Projects Completed (major projects)
- Gives suggestions for improvement if the prediction is negative.

## Demo
![Demo Screenshot](#) <!-- Add a screenshot if available -->

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd college-placement-predictor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app/app.py
   ```
4. **Open the app:**
   - Visit the local URL provided by Streamlit (usually http://localhost:8501) in your browser.

## File Structure
- `app/app.py`: Main Streamlit app.
- `app/placement_logistic_model.pkl`, `app/scaler.pkl`: Trained model and scaler.
- `app/college_student_placement_dataset.csv`: Dataset for statistics and suggestions.
- `notebook/model_training.ipynb`: Model training and evaluation notebook.
- `requirements.txt`: Python dependencies.

## Disclaimer
> ⚠️ This is a predictive tool for learning purposes. Actual placement depends on multiple real-life factors not captured by this model.

## License
[MIT](LICENSE) <!-- Update if you add a license file -->