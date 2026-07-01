# data_register.py
# Creates the Hugging Face dataset repo and uploads the raw tourism.csv file.

from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# ── Config ────────────────────────────────────────────────────
HF_TOKEN  = os.environ["HF_TOKEN"]
HF_USER   = "kavumadouglas"
repo_id   = f"{HF_USER}/tourism"
DATA_PATH = "tourism_project/data/tourism.csv"

api = HfApi()

# ── 1. Create dataset repository on Hugging Face Hub ─────────
try:
    create_repo(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    print(f"✅ Dataset repo ready: https://huggingface.co/datasets/{repo_id}")
except HfHubHTTPError as e:
    print(f"❌ Failed to create repo: {e}")
    raise

# ── 2. Upload tourism.csv ─────────────────────────────────────
try:
    api.upload_file(
        path_or_fileobj=DATA_PATH,
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"✅ tourism.csv uploaded to: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    raise

print("\n✅ Step 2 (Data Registration) complete.")
