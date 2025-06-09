import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up the Streamlit app layout
st.set_page_config(page_title="Heart Attack Prediction", layout="centered")

st.title("Heart Attack Prediction using Random Forest")
st.markdown("""
Predict the likelihood of a heart attack based on your clinical parameters.  
Fill out the form below and click **Predict** to see your risk.
""")

@st.cache_data
def load_and_train_model():
    # Load your local dataset (make sure the file is in the same directory)
    data = pd.read_csv("heart_attack_prediction_dataset.csv")

    # Check that the 'target' column exists
    if 'target' not in data.columns:
        st.error("❌ The dataset does not contain a 'target' column.")
        st.stop()

    # Split features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X.columns

# Train and load model
model, accuracy, feature_names = load_and_train_model()

# Display model accuracy
st.markdown(f"**Model Accuracy on Test Data:** {accuracy:.2%}")

# Collect user input
st.header("Enter Your Health Parameters:")

age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=230)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2])
ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# Prepare input for prediction
input_df = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == "Male" else 0],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
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
        st.write(f"**{feature.capitalize()}**: {input_df.iloc[0][feature]}")

st.markdown("---")
st.markdown("**Note:** This tool is for educational purposes only and not a substitute for medical advice.")
