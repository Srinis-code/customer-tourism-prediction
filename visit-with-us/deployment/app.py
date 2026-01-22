%%writefile visit-with-us/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ---------------------------
# Load model
# ---------------------------
model_path = hf_hub_download(
    repo_id="ksricheenu/customer-tourism-prediction-artifacts",
    filename="best_tourism_targeting_model_v1.joblib",
    repo_type="model"
)

model = joblib.load(model_path)

st.title("Customer Tourism Prediction")

# ---------------------------
# User Inputs
# ---------------------------
Age = st.number_input("Age", 15, 80, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
DurationOfPitch = st.number_input("Duration of Pitch", 5, 180, 30)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfFollowups = st.number_input("Number Of Followups", 1, 10, 2)
ProductPitched = st.selectbox("Product Pitched", ["Super Deluxe", "Deluxe", "Basic", "Standard"])
PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", 0, 100, 1)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)

# ---------------------------
# Assemble EXACT training schema
# ---------------------------
input_data = pd.DataFrame([{
    "CustomerID": 0,
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": 1,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": 2,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 0,
    "NumberOfChildrenVisiting": 0,
    "Designation": "Executive",
    "MonthlyIncome": 50000,
    "Unnamed: 0": 0
}])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Customer Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer WILL purchase the package" if prediction == 1 else "Customer will NOT purchase the package"
    st.success(result)
