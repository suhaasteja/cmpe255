import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved models
compensation_model = joblib.load('compensation_model.pkl')
stress_model = joblib.load('stress_model.pkl')
job_title_encoder = joblib.load('job_title_encoder.pkl')
stress_encoder = joblib.load('stress_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df_combined = pd.read_csv('df_combined.csv')

# Helper function to get trend data
def get_trend_data(job_title):
    filtered_data = df_combined[df_combined['Job Title'] == job_title]
    trend_data = filtered_data.groupby('Year').agg(
        Average_Compensation=('Total Cash Compensation', 'mean'),
        Average_Stress_Level=('Stress Level', lambda x: x.mode()[0] if len(x) > 0 else None)
    ).reset_index()
    return trend_data

# UI Title
st.title("Job Compensation and Stress Level Predictor with Trend Charts")

# User inputs for the prediction
st.header("Input Job Details")

# Select job title and year for prediction
job_title = st.selectbox("Select Job Title", df_combined['Job Title'].unique())
year = st.number_input("Enter Year", min_value=2000, max_value=2030, value=2024)

# Prediction button
if st.button("Predict"):
    # Encode the job title input
    job_title_encoded = job_title_encoder.transform([job_title])[0]
    
    # Prepare input data
    input_data = np.array([[job_title_encoded, year]])
    scaled_input = scaler.transform(input_data)
    
    # Make predictions
    predicted_comp = compensation_model.predict(scaled_input)[0]
    stress_pred = stress_model.predict(scaled_input)[0]
    stress_label = stress_encoder.inverse_transform([stress_pred])[0]
    
    # Display predictions
    st.subheader("Prediction Results")
    st.write(f"**Predicted Total Compensation**: ${predicted_comp:,.2f}")
    st.write(f"**Predicted Stress Level**: {stress_label}")

# Now show the trend charts
st.header(f"Trend Charts for {job_title}")

# Get real trend data
trend_data = get_trend_data(job_title)

# Compensation trend chart
st.subheader(f"Compensation Trend Over the Years for {job_title}")
fig, ax = plt.subplots()
ax.plot(trend_data['Year'], trend_data['Average_Compensation'], marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Average Compensation ($)")
ax.set_title(f"Average Compensation Over Time for {job_title}")
st.pyplot(fig)

# Stress level trend chart
st.subheader(f"Stress Level Trend Over the Years for {job_title}")
fig, ax = plt.subplots()
stress_counts = trend_data['Average_Stress_Level'].value_counts()
ax.bar(stress_counts.index, stress_counts.values)
ax.set_xlabel("Stress Level")
ax.set_ylabel("Count")
ax.set_title(f"Stress Levels Over Time for {job_title}")
st.pyplot(fig)
