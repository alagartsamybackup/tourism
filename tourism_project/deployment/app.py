import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

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

# Get Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Load model and preprocessing objects from Hugging Face
@st.cache_resource
def load_model():
    try:
        # Download model files from Hugging Face
        model_path = hf_hub_download(
            repo_id="alagarst/tourism-package-prediction-model",
            filename="best_model.pkl",
            token=HF_TOKEN # Pass token for authentication
        )
        scaler_path = hf_hub_download(
            repo_id="alagarst/tourism-package-prediction-model",
            filename="scaler.pkl",
            token=HF_TOKEN # Pass token for authentication
        )
        label_encoders_path = hf_hub_download(
            repo_id="alagarst/tourism-package-prediction-model",
            filename="label_encoders.pkl",
            token=HF_TOKEN # Pass token for authentication
        )
        imputer_numeric_path = hf_hub_download(
            repo_id="alagarst/tourism-package-prediction-model",
            filename="imputer_numeric.pkl",
            token=HF_TOKEN # Pass token for authentication
        )
        imputer_categorical_path = hf_hub_download(
            repo_id="alagarst/tourism-package-prediction-model",
            filename="imputer_categorical.pkl",
            token=HF_TOKEN # Pass token for authentication
        )
        # Download training column names
        X_train_columns_path = hf_hub_download(
            repo_id="alagarst/tourism-package-prediction-dataset", # Columns are saved in the dataset repo
            filename="X_train_columns.pkl",
            token=HF_TOKEN # Pass token for authentication
        )


        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(label_encoders_path)
        imputer_numeric = joblib.load(imputer_numeric_path)
        imputer_categorical = joblib.load(imputer_categorical_path)
        X_train_columns = joblib.load(X_train_columns_path)


        return model, scaler, label_encoders, imputer_numeric, imputer_categorical, X_train_columns
    except Exception as e:
        st.error(f"Error loading model or preprocessing objects: {e}")
        return None, None, None, None, None, None

if HF_TOKEN is None:
    st.error("Hugging Face token not found. Please set the HF_TOKEN environment variable in your Hugging Face Space settings.")
    model, scaler, label_encoders, imputer_numeric, imputer_categorical, X_train_columns = None, None, None, None, None, None
else:
    model, scaler, label_encoders, imputer_numeric, imputer_categorical, X_train_columns = load_model()


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
if model is not None and scaler is not None and label_encoders is not None and X_train_columns is not None:
    # Ensure input_df has the same columns as X_train and in the same order
    # Add missing columns with default values (e.g., 0 or median/mode if applicable)
    # For simplicity here, we'll assume the input form covers all necessary features
    # and focus on reindexing to match the training column order.
    # A more robust approach might involve imputing missing columns with training data statistics.

    # Reindex input_df to match the order of X_train columns
    input_df = input_df.reindex(columns=X_train_columns, fill_value=0) # Using 0 as a placeholder, actual imputation is handled by imputers if needed

    # Explicitly drop 'Unnamed: 0' column if it exists (it should now exist after reindexing if it was in X_train_columns)
    if 'Unnamed: 0' in input_df.columns:
        input_df = input_df.drop('Unnamed: 0', axis=1)

    # Impute any potentially missing values after reindexing (though the form should cover most)
    # This step might be redundant if the form covers all features, but is safer.
    numeric_cols_input = input_df.select_dtypes(include=[np.number]).columns
    categorical_cols_input = input_df.select_dtypes(include=['object']).columns

    if imputer_numeric:
         input_df[numeric_cols_input] = imputer_numeric.transform(input_df[numeric_cols_input])
    if imputer_categorical:
         input_df[categorical_cols_input] = imputer_categorical.transform(input_df[categorical_cols_input])


    # Encode categorical variables
    for column in categorical_cols_input:
        if column in label_encoders:
            # Ensure the value is in the known classes, handle unknown values if necessary
            try:
                # We need to transform the entire column (even if it's just one row)
                input_df[column] = label_encoders[column].transform(input_df[column].astype(str))
            except ValueError:
                st.warning(f"Unknown value for {column}: {input_df[column].iloc[0]}. Using mode imputation.")
                # Use the categorical imputer for unseen values - note: imputer_categorical.transform expects 2D array
                input_df[column] = imputer_categorical.transform(input_df[[column]])[:,0]


    # Scale features
    # Ensure the order of columns matches training data AFTER dropping 'Unnamed: 0' and encoding
    # This should be handled by the reindex step and dropping 'Unnamed: 0'
    # Let's get the columns after dropping 'Unnamed: 0' for the scaler transform
    cols_for_scaling = [col for col in X_train_columns if col != 'Unnamed: 0']
    input_df_scaled = input_df[cols_for_scaling]


    input_scaled = scaler.transform(input_df_scaled)


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
    st.error("Model or preprocessing objects not loaded properly. Please check the configuration.")

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
