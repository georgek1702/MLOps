from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Initialize API with your token (make sure HF_TOKEN is set)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define your Space details
repo_id = "Georgek17/Customer_Visit_Predictor"
repo_type = "space"
folder_path = "tourism_project/deployment"

# Step 1: Check if the Space exists
# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload your app files to the Space
print(f"Uploading folder '{folder_path}' to Hugging Face Space...")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""
)
print("Upload complete.")
