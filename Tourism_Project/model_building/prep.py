# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
#Load dataset from HUggingface space
DATASET_PATH = "hf://datasets/Georgek17/Customer_Visit_Predictor/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Cleaning data

# Drop Unnecessary Columns
if "CustomerID" in tourism_dataset.columns:
    tourism_dataset = tourism_dataset.drop(columns=["CustomerID"])

# Handle Missing Values

# List of numerical features in the dataset
numeric_features = [
    'Age',                     # Customer's age
    'CityTier',                # The city category based on development, population, and living standards (Tier 1 > Tier 2 >
    'DurationOfPitch',         # Duration of the sales pitch delivered to the customer.
    'NumberOfPersonVisiting',  # Total number of people accompanying the customer on the trip.
    'NumberOfFollowups',       # Total number of follow-ups by the salesperson after the sales pitch 
    'PreferredPropertyStar',   # Preferred hotel rating by the customer.
    'NumberOfTrips',               # Average number of trips the customer takes annually.
    'Passport',                    # Whether the customer holds a valid passport (0: No, 1: Yes).
    'PitchSatisfactionScore',      # Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar',                      # Whether the customer owns a car (0: No, 1: Yes).
    'NumberOfChildrenVisiting',    # Number of children below age 5 accompanying the customer.
    'MonthlyIncome'                # Gross monthly income of the customer.
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The method by which the customer was contacted
    'Occupation',            # Customer's occupation (e.g., Salaried, Freelancer).
    'Gender',                # Gender of the customer
    'ProductPitched',        # The type of product pitched to the customer.
    'MaritalStatus',         # Marital status of the customer (Single, Married, Divorced).
    'Designation'            # Customer's designation in their current organization.
]

# Fill numeric columns with median
for col in numeric_features:
    if col in tourism_dataset.columns:
        tourism_dataset[col] = tourism_dataset[col].fillna(tourism_dataset[col].median())

# Fill categorical columns with mode
for col in categorical_features:
    if col in tourism_dataset.columns:
        tourism_dataset[col] = tourism_dataset[col].fillna(tourism_dataset[col].mode()[0])

# Drop rows where target variable is missing
if "ProdTaken" in tourism_dataset.columns:
    tourism_dataset = tourism_dataset.dropna(subset=["ProdTaken"])

# Remove Duplicates
tourism_dataset = tourism_dataset.drop_duplicates()

# Encode Categorical Columns will be done within training

# Define the target variable for the classification task
target = 'ProdTaken'


# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset.drop(columns=['ProdTaken'])

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.25,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

#Upload the resulting train and test datasets back to the Hugging Face data space.
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Georgek17/Customer_Visit_Predictor",
        repo_type="dataset",
    )
