from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Initialize API with your Hugging Face token
api = HfApi(token=os.getenv("HF_TOKEN"))  # or replace with your actual token for testing

# Define your Space details
repo_id = "Georgek17/Customer_Visit_Predictor"  # your Hugging Face Space name
repo_type = "space"
space_sdk = "streamlit"   # since you’re using Streamlit
folder_path = "tourism_project/deployment"  # folder containing your app files

# Step 1: Check if the Space already exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        space_sdk=space_sdk  # required for Spaces
    )
    print(f"Space '{repo_id}' created successfully.")

# Step 2: Upload your app files to the Space
print(f"Uploading folder '{folder_path}' to Hugging Face Space...")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""
)
print("Upload complete. Your Streamlit app is live on Hugging Face.")
