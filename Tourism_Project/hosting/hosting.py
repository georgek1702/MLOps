from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError
import os

# Initialize API with your token (make sure HF_TOKEN is set)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define your Space details
repo_id = "Georgek17/Customer_Visit_Predictor"
repo_type = "space"
folder_path = "tourism_project/deployment"

# Step 1: Check if the Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new Space...")
    create_repo(
        name="Customer_Visit_Predictor",
        repo_type="space",
        private=False,
        exist_ok=True
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
print("Upload complete.")
