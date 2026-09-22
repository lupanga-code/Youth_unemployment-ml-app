import streamlit as st
import pandas as pd
import joblib

# Cache model loading
@st.cache_data
def load_model():
    return joblib.load("youth_employment_model.pkl")  # badilisha jina la file kulingana na model yako

model = load_model()

# Cache data loading (optional kama unatumia dataset kubwa)
@st.cache_data
def load_data():
    return pd.read_csv("youth_unemployment_dataset.csv")  # badilisha jina la file kulingana na dataset yako

# Title ya app
st.title("Youth Unemployment Prediction App")

# Input fields
age = st.number_input("Enter Age", min_value=15, max_value=60)
education = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary"])
experience = st.number_input("Years of Experience", min_value=0, max_value=40)

# Encode education level
education_map = {"Primary": 0, "Secondary": 1, "Tertiary": 2}
education_encoded = education_map[education]

# Prepare data for prediction
input_data = pd.DataFrame([[age, education_encoded, experience]],
                          columns=["Age", "Education", "Experience"])

# Prediction button
if st.button("Predict Employment Status"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Employment Status: {prediction[0]}")
