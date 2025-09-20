import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Georgek17/vistit-predictor-model", filename="best_visit_predictor_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Customer visit Prediction App")
st.write("The Customer visit Prediction App is an internal tool for predicts whether customer will purchase the newly introduced Wellness Tourism Package before contacting them based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to purchase the Wellness Tourism Package.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact (method by which the customer was contacted)", ["Self Enquiry", "Company Invited"])
CityTier= st.selectbox("City Tier (The city category based on development, population, and living standards)", ["1", "2", "3"])
DurationOfPitch = st.number_input("DurationOfPitch (Duration of the sales pitch delivered to the customer.)", min_value=1, value=14)
Occupation= st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
Gender= st.selectbox("Gender", ["Female", "Male"])
NumberOfPersonVisiting= st.number_input("Number Of PersonVisiting (Total number of people accompanying the customer on the trip.)", value=3)
NumberOfFollowups= st.number_input("Number Of Followups (Total number of follow-ups by the salesperson after the sales pitch.)", value=3)
ProductPitched= st.selectbox("Product Pitched (The type of product pitched to the customer.)", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
PreferredPropertyStar= st.selectbox("Preferred Property Star (Preferred hotel rating by the customer.)", ["1", "2", "3", "4", "5"])
MaritalStatus= st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Unmarried"])
NumberOfTrips= st.number_input("Number Of Trips (Average number of trips the customer takes annually.)", min_value=1, value=2)
Passport= st.selectbox("Has Passport? (Whether the customer holds a valid passport (0: No, 1: Yes).)", ["0", "1"])
PitchSatisfactionScore= st.selectbox("Pitch Satisfaction Score (Score indicating the customer's satisfaction with the sales pitch.)", ["1", "2", "3", "4", "5"])
OwnCar= st.selectbox("Own Car? (Whether the customer owns a car (0: No, 1: Yes).)", ["0", "1"])
NumberOfChildrenVisiting= st.number_input("Number Of Children Visiting)", value=1)
Designation= st.selectbox("Designation (Customer's designation in their current organization.)", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income (Gross monthly income of the customer.)", min_value=0, value=1700)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups':NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar' : PreferredPropertyStar,
    'MaritalStatus' : MaritalStatus,
    'NumberOfTrips' : NumberOfTrips, 
    'Passport' : Passport,
    'PitchSatisfactionScore' : PitchSatisfactionScore,
    'OwnCar' : OwnCar,
    'NumberOfChildrenVisiting' : NumberOfChildrenVisiting,
    'Designation' : Designation,
    'MonthlyIncome' : MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the Wellness Tourism Package " if prediction == 1 else "not purchase the Wellness Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
