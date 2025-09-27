from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Georgek17/Customer_Visit_Predictor"
repo_type = "space"
space_sdk = "streamlit"
folder_path = "tourism_project/deployment"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        space_sdk=space_sdk
    )
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""
)
print("Upload complete.")
