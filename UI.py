import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# Page configuration
st.set_page_config(
    page_title="Employee Compensation & Stress Predictor",
    page_icon="💼",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    try:
        return {
            'compensation': joblib.load('compensation_model.pkl'),
            'stress': joblib.load('stress_model.pkl'),
            'job_encoder': joblib.load('job_title_encoder.pkl'),
            'stress_encoder': joblib.load('stress_encoder.pkl'),
            'scaler': joblib.load('scaler.pkl')
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('df_filtered.csv')
        inflation_data = pd.read_excel('inflationData.xlsx', skiprows=11)
        
        # Clean inflation data
        inflation_data.columns = inflation_data.iloc[0]
        inflation_data = inflation_data.iloc[1:].reset_index(drop=True)
        inflation_data['Year'] = pd.to_numeric(inflation_data['Year'], errors='coerce')
        inflation_data['Average Inflation'] = inflation_data.iloc[:, 1:].mean(axis=1)
        
        return df, inflation_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Initialize models and data
models = load_models()
df, inflation_data = load_data()
poly = PolynomialFeatures(degree=2)

# Title and description
st.title("🎯 Enhanced Employee Compensation & Stress Predictor")
st.markdown("""
    This advanced tool predicts employee compensation and stress levels using machine learning models
    trained on historical data, incorporating economic factors like inflation.
""")

if models and df is not None and inflation_data is not None:
    # Sidebar inputs
    with st.sidebar:
        st.header("📝 Input Parameters")
        
        job_title = st.selectbox(
            "Select Job Title",
            sorted(df['Job Title'].unique()),
            help="Choose the job title for prediction"
        )
        
        # Year selection with future projection warning
        current_year = 2024
        year = st.slider(
            "Select Prediction Year",
            min_value=2020,
            max_value=2030,
            value=current_year,
            help="Select the year for prediction. Future predictions may have increased uncertainty."
        )
        
        # Custom inflation input for future predictions
        if year > 2023:
            custom_inflation = st.slider(
                "Expected Inflation Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Enter expected inflation rate for future predictions"
            )
        
        # Analysis options
        st.header("📊 Analysis Options")
        show_historical = st.checkbox("Show Historical Trends", value=True)
        show_compensation_distribution = st.checkbox("Show Compensation Distribution", value=True)
        show_stress_analysis = st.checkbox("Show Stress Analysis", value=True)

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🎯 Prediction Results")
        
        if st.button("Generate Prediction"):
            with st.spinner('Analyzing data and generating predictions...'):
                try:
                    # Prepare input data
                    job_encoded = models['job_encoder'].transform([job_title])[0]
                    
                    # Get inflation rate
                    if year <= 2023:
                        inflation = inflation_data[inflation_data['Year'] == year]['Average Inflation'].iloc[0]
                    else:
                        inflation = custom_inflation / 100  # Convert percentage to decimal
                    
                    # Calculate trend using polynomial features
                    year_array = np.array([[year]])
                    year_poly = poly.fit_transform(year_array)
                    trend_pred = year_poly[0][1]  # Using linear term
                    
                    # Create input for scaling (only job_encoded and inflation)
                    input_for_scaling = np.array([[job_encoded, inflation]])
                    scaled_features = models['scaler'].transform(input_for_scaling)
                    
                    # Create final input by combining scaled features with year and trend
                    input_data = np.hstack((
                        scaled_features,
                        np.array([[year, trend_pred]])
                    ))
                    
                    # Make predictions
                    comp_pred = models['compensation'].predict(input_data)[0]
                    stress_pred = models['stress'].predict(input_data)[0]
                    stress_label = models['stress_encoder'].inverse_transform([stress_pred])[0]
                    
                    # Display predictions
                    col_comp1, col_comp2 = st.columns(2)
                    with col_comp1:
                        st.metric("Predicted Base Compensation", f"${comp_pred:,.2f}")
                    with col_comp2:
                        st.metric("Inflation-Adjusted", f"${comp_pred * (1 + inflation):,.2f}")
                    
                    st.info(f"Predicted Stress Level: **{stress_label}**")
                    
                    # Display additional insights
                    st.markdown("### Additional Insights")
                    col_insights1, col_insights2 = st.columns(2)
                    with col_insights1:
                        st.metric("Inflation Rate", f"{inflation:.1%}")
                    with col_insights2:
                        baseline = df[df['Job Title'] == job_title]['Total Cash Compensation'].mean()
                        change = ((comp_pred - baseline) / baseline) * 100
                        st.metric("vs. Historical Average", f"{change:+.1f}%")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.error("Debug info:")
                    st.write({
                        "job_encoded": job_encoded,
                        "inflation": inflation,
                        "year": year,
                        "trend_pred": trend_pred,
                        "scaled_features shape": scaled_features.shape,
                        "input_data shape": input_data.shape
                    })
    
    # Display visualizations
    if show_historical or show_compensation_distribution or show_stress_analysis:
        with col2:
            job_data = df[df['Job Title'] == job_title].copy()
            
            if show_historical:
                st.subheader("📈 Historical Trends")
                fig_comp = plt.figure(figsize=(10, 6))
                sns.lineplot(data=job_data, x='Year', y='Total Cash Compensation')
                plt.title(f'Historical Compensation\n{job_title}')
                plt.ylabel('Compensation ($)')
                st.pyplot(fig_comp)
                plt.close()
            
            if show_compensation_distribution:
                st.subheader("💰 Compensation Distribution")
                fig_dist = plt.figure(figsize=(10, 4))
                sns.histplot(data=job_data, x='Total Cash Compensation', bins=20)
                plt.title('Historical Distribution')
                st.pyplot(fig_dist)
                plt.close()
            
            if show_stress_analysis:
                st.subheader("😰 Stress Analysis")
                stress_counts = job_data['Stress Level'].value_counts()
                fig_stress = plt.figure(figsize=(8, 4))
                plt.pie(stress_counts, labels=stress_counts.index, autopct='%1.1f%%')
                plt.title('Stress Level Distribution')
                st.pyplot(fig_stress)
                plt.close()

    # Footer
    st.markdown("---")
    st.markdown("""
        **Note:** Predictions are based on historical data and current trends. Future predictions
        should be used as guidance only, as they may not account for unforeseen market changes.
    """)

else:
    st.error("Unable to load required components. Please check the installation and data files.")