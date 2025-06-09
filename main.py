# heart_attack_prediction_streamlit.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(page_title="Heart Attack Prediction", layout="centered")

st.title("Heart Attack Prediction using Random Forest")
st.markdown(
    """
    Predict the likelihood of a heart attack based on your clinical parameters.
    Fill out the form below and click **Predict** to see your risk.
    """
)

@st.cache_data
def load_and_train_model():
    # Load dataset directly from GitHub (heart.csv from a reliable source)
    url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
    data = pd.read_csv(url)

    # Split features and target variable
    X = data.drop('target', axis=1)
    y = data['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X.columns

# Load and train the model once
model, accuracy, feature_names = load_and_train_model()

st.markdown(f"**Model Accuracy on Test Data:** {accuracy:.2%}")

# User input form for prediction
st.header("Enter Your Health Parameters:")

age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=230)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], help="0 = False, 1 = True")
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], help="0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy")
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], help="0 = No, 1 = Yes")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], help="0=upsloping, 1=flat, 2=downsloping")
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4], help="0 to 4")
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], help="1=normal, 2=fixed defect, 3=reversible defect")

# Prepare input for prediction
input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [1 if sex == "Male" else 0],
    'CP': [cp],
    'Trestbps': [trestbps],
    'Chol': [chol],
    'Fbs': [fbs],
    'Restecg': [restecg],
    'Thalach': [thalach],
    'Exang': [exang],
    'Oldpeak': [oldpeak],
    'Slope': [slope],
    'Ca': [ca],
    'Thal': [thal]
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart attack predicted with confidence {prediction_proba:.1%}. Please consult a doctor.")
    else:
        st.success(f"❤️ Low risk of heart attack predicted with confidence {prediction_proba:.1%}.")

    st.markdown("### Your Input Summary:")
    for feature in input_df.columns:
        st.write(f"**{feature}**: {input_df.iloc[0][feature]}")

st.markdown("---")
st.markdown(
    """
    **Note:** This prediction is for educational purposes only and is not a substitute for professional medical advice.
    Always consult a healthcare professional for medical concerns.
    """
)
