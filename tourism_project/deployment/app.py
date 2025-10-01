import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Set page config
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# Title and description
st.title("🏖️ Tourism Package Prediction")
st.markdown("""
This app predicts whether a customer will purchase the Wellness Tourism Package 
based on their profile and interaction data.
""")

# Load model and preprocessing objects from Hugging Face
@st.cache_resource
def load_model():
    try:
        # Download model files from Hugging Face
        model_path = hf_hub_download(
            repo_id="your_username/tourism-package-prediction-model",
            filename="best_model.pkl"
        )
        scaler_path = hf_hub_download(
            repo_id="your_username/tourism-package-prediction-model",
            filename="scaler.pkl"
        )
        label_encoders_path = hf_hub_download(
            repo_id="your_username/tourism-package-prediction-model",
            filename="label_encoders.pkl"
        )
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(label_encoders_path)
        
        return model, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, label_encoders = load_model()

# Create input form
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=18, max_value=80, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    
with col2:
    num_persons = st.slider("Number of Persons Visiting", min_value=1, max_value=5, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    num_trips = st.slider("Number of Trips per Year", min_value=1, max_value=10, value=2)
    
with col3:
    passport = st.selectbox("Passport", [0, 1])
    own_car = st.selectbox("Own Car", [0, 1])
    num_children = st.slider("Number of Children Visiting", min_value=0, max_value=3, value=0)
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
    monthly_income = st.slider("Monthly Income", min_value=10000, max_value=100000, value=30000, step=1000)

st.header("Interaction Details")

col4, col5, col6 = st.columns(3)

with col4:
    type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Enquiry'])
    duration_pitch = st.slider("Duration of Pitch (minutes)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
    
with col5:
    num_followups = st.slider("Number of Follow-ups", min_value=1, max_value=6, value=3)
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    
with col6:
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

# Prepare input data
input_data = {
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_persons,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input data
if model is not None and scaler is not None and label_encoders is not None:
    # Encode categorical variables
    for column in input_df.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            input_df[column] = label_encoders[column].transform([input_df[column].iloc[0]])[0]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    if st.button("Predict Purchase Probability"):
        prediction_proba = model.predict_proba(input_scaled)[0]
        prediction = model.predict(input_scaled)[0]
        
        st.header("Prediction Results")
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.metric("Probability of Purchase", f"{prediction_proba[1]:.2%}")
            
        with col8:
            if prediction == 1:
                st.success("🎯 Recommended: This customer is likely to purchase the package!")
            else:
                st.warning("⚠️ Not Recommended: This customer is unlikely to purchase the package.")
        
        # Show probability breakdown
        st.subheader("Probability Breakdown")
        prob_df = pd.DataFrame({
            'Outcome': ['Will Not Purchase', 'Will Purchase'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        st.bar_chart(prob_df.set_index('Outcome'))

else:
    st.error("Model not loaded properly. Please check the configuration.")

# Add some information about the model
with st.expander("About this Model"):
    st.markdown("""
    This machine learning model predicts the likelihood of a customer purchasing 
    the Wellness Tourism Package based on various factors including:
    
    - Customer demographics (age, income, occupation)
    - Travel preferences (property star, number of trips)
    - Interaction history (pitch duration, follow-ups)
    
    The model was trained using ensemble methods and achieves high accuracy 
    in identifying potential customers for targeted marketing campaigns.
    """)
