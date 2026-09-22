import streamlit as st
import joblib
import pandas as pd

# Cache model loading
@st.cache_data
def load_model():
    return joblib.load("youth_employment_model.pkl")

# Cache data loading (kama unatumia dataset kubwa)
@st.cache_data
def load_data():
    return pd.read_csv("youth_unemployment_dataset.csv")   # badilisha jina la file kulingana na dataset yako

# Load trained model
model = joblib.load("model.pkl")

st.title("Youth Unemployment Prediction App")

# User input form
age = st.number_input("Age", min_value=15, max_value=60, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary"])
skills = st.selectbox("Skills", ["None", "Basic", "Advanced"])
experience = st.number_input("Work Experience (years)", min_value=0, max_value=40)

# Convert input to DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education": [education],
    "Skills": [skills],
    "Experience": [experience]
})

# Predict
prediction = model.predict(input_data)[0]

st.write("### Employment Status:")
st.write("✅ Employed" if prediction == 1 else "❌ Unemployed")
