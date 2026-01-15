import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="ksricheenu/customer-tourism-prediction-model", filename="best_tourism_targeting_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Customer Tourism Prediction System")
st.write("""
This application predicts the customers who are likely to buy the Tourism package 
Please enter the Customer related information to get a prediction of buying Tourism Package.
""")

# User input
Age = st.number_input("Age", min_value=15.0, max_value=80.0, value=15.0)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=5.0, max_value=180.0, value=1.0)
Occupation  = st.selectbox("Occupation", ["Salaried", "Free Lancer"])
Gender  = st.selectbox("Gender", ["Male", "Female"])
NumberOfFollowups = st.number_input("Number Of Followups", min_value=1.0, max_value=10.0, value=1.0)
ProductPitched = st.selectbox("Product Pitched", ["Super Deluxe", "Deluxe", "Basic", "Standard"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1.0, max_value=5.0, value=1.0)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=1.0, max_value=100.0, value=1.0)
passport_input = st.selectbox("Passport", ["No", "Yes"])
passport = 1 if passport_input == "Yes" else 0
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=1.0, max_value=10.0, value=1.0)
PitchSatisfactionScore  = st.number_input("Pitch Satisfaction Score", min_value=1.0, max_value=5.0, value=1.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'Type of Contact': TypeofContact,
    'Duration of Pitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'Number Of Followups': NumberOfFollowups,
    'Product Pitched': ProductPitched,
    'Preferred Property Star': PreferredPropertyStar,
    'Marital Status': MaritalStatus,
    'Number Of Trips': NumberOfTrips,
    'Passport': passport,
    'Number Of Children Visiting': NumberOfChildrenVisiting,
    'Pitch Satisfaction Score': PitchSatisfactionScore
}])


if st.button("Predict Customer Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase" if prediction == 1 else "Customer will NOT purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

