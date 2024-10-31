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
df_filtered = pd.read_csv('df_filtered.csv')

# Helper function to get trend data
def get_trend_data(job_title):
    # Convert stress levels to numerical values: low=0, medium=1, high=2
    stress_mapping = {'Low Stress': 0, 'Medium Stress': 50, 'High Stress': 100}
    df_filtered['Stress Level Numeric'] = df_filtered['Stress Level'].map(stress_mapping)
    
    filtered_data = df_filtered[df_filtered['Job Title'] == job_title]
    
    trend_data = filtered_data.groupby('Year').agg(
        Average_Compensation=('Total Cash Compensation', 'mean'),
        Average_Stress_Level=('Stress Level Numeric', 'mean')  # Now using average of numeric stress levels
    ).reset_index()
    
    return trend_data

# UI Title
st.title("Job Compensation and Stress Level Predictor with Trend Charts")

# User inputs for the prediction
st.header("Input Job Details")

# Select job title and year for prediction
job_title = st.selectbox("Select Job Title", df_filtered['Job Title'].unique())
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
ax.plot(trend_data['Year'], trend_data['Average_Stress_Level'], marker='o', color='orange')
ax.set_xlabel("Year")
ax.set_ylabel("Average Stress Level Percentage")
ax.set_ylim(0, 100)
ax.set_title(f"Average Stress Level Over Time for {job_title}")
st.pyplot(fig)
