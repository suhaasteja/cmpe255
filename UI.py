import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved models
compensation_model = joblib.load('compensation_model.pkl')
stress_model = joblib.load('stress_model.pkl')

# Load dataset to get job title information
df = pd.read_csv('df_combined.csv')

# Encode job title for user input
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Job Title Encoded'] = label_encoder.fit_transform(df['Job Title'])

# UI Title
st.title("Job Compensation and Stress Level Prediction")

# User inputs for the prediction
st.header("Input Job Details")

# Select job title, base pay, overtime, etc.
job_title = st.selectbox("Select Job Title", df['Job Title'].unique())
base_pay = st.number_input("Enter Base Pay", min_value=0, value=50000, step=1000)
overtime = st.number_input("Enter Overtime Pay", min_value=0, value=5000, step=100)
hourly_rate = st.number_input("Enter Hourly Rate", min_value=0, value=30, step=1)
hours_worked = st.number_input("Enter Hours Worked Overtime", min_value=0, value=100, step=1)
year = st.number_input("Enter Year", min_value=2000, max_value=2025, value=2023)

# Prepare input data for the models
job_title_encoded = label_encoder.transform([job_title])[0]
input_data = np.array([[job_title_encoded, base_pay, overtime, hourly_rate, hours_worked, year]])

# Prediction button
if st.button("Predict"):
    # Predict compensation using the RandomForestRegressor
    compensation_prediction = compensation_model.predict(input_data)[0]
    
    # Predict stress level using the RandomForestClassifier
    stress_prediction_encoded = stress_model.predict(input_data)[0]
    stress_prediction = label_encoder.inverse_transform([stress_prediction_encoded])[0]
    
    # Display predictions
    st.subheader("Prediction Results")
    st.write(f"**Predicted Total Compensation**: ${compensation_prediction:,.2f}")
    st.write(f"**Predicted Stress Level**: {stress_prediction}")

# Plot trends (optional, if you want to include trend graphs)
st.header("Trends")

# Average Compensation Trend over the years
st.subheader("Average Compensation Over the Years")
compensation_trend = df.groupby('Year')['Total Cash Compensation'].mean()

st.line_chart(compensation_trend)

# Average Stress Level Trend over the years
st.subheader("Average Stress Level Over the Years")
df['Stress Level Encoded'] = label_encoder.fit_transform(df['Stress Level'])
stress_trend = df.groupby('Year')['Stress Level Encoded'].mean()

st.line_chart(stress_trend)
