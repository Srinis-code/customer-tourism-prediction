from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "ksricheenu/customer-tourism-prediction-dataset"
repo_type = "dataset"

# Step 1: Check if the dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Path relative to GitHub repo root
folder_path = "visit-with-us/data"

# Upload the folder
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset upload completed successfully.")



