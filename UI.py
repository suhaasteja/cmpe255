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

# UI Title
st.title("Job Compensation and Stress Level Predictor with Trend Charts")

# User inputs for the prediction
st.header("Input Job Details")

# Select job title and year for prediction
job_title = st.selectbox("Select Job Title", job_title_encoder.classes_)
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

# Trend data (example, you can replace it with real data or dynamically generate it)
trend_data = pd.DataFrame({
    'Year': np.arange(2015, 2024),
    'Average Compensation': np.random.uniform(400000, 600000, 9),  # Simulated data
    'Average Stress Level': np.random.choice(['Low Stress', 'Medium Stress', 'High Stress'], 9)
})

# Compensation trend chart
st.subheader(f"Compensation Trend Over the Years for {job_title}")
fig, ax = plt.subplots()
ax.plot(trend_data['Year'], trend_data['Average Compensation'], marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Average Compensation ($)")
ax.set_title(f"Average Compensation Over Time for {job_title}")
st.pyplot(fig)

# Stress level trend chart
st.subheader(f"Stress Level Trend Over the Years for {job_title}")
fig, ax = plt.subplots()
stress_counts = trend_data['Average Stress Level'].value_counts()
ax.bar(stress_counts.index, stress_counts.values)
ax.set_xlabel("Stress Level")
ax.set_ylabel("Count")
ax.set_title(f"Stress Levels Over Time for {job_title}")
st.pyplot(fig)
