# hosting.py
# Creates a Hugging Face Docker Space and uploads all deployment
# files to it: README.md, Dockerfile, app.py, requirements.txt

from huggingface_hub import HfApi, create_repo
import os

# ── Config ────────────────────────────────────────────────────
HF_TOKEN       = os.environ["HF_TOKEN"]
HF_USER        = "kavumadouglas"
SPACE_NAME     = "tourism-prediction-app"
space_repo_id  = f"{HF_USER}/{SPACE_NAME}"
DEPLOYMENT_DIR = "tourism_project/deployment"

api = HfApi()

# ── 1. Create the Hugging Face Docker Space ───────────────────
create_repo(
    repo_id=space_repo_id,
    repo_type="space",
    space_sdk="docker",
    token=HF_TOKEN,
    exist_ok=True
)
print(f"✅ Space ready: https://huggingface.co/spaces/{space_repo_id}")

# ── 2. Upload all deployment files ────────────────────────────
# README.md is uploaded first — it sets app_port: 8501 so HF
# knows which port to health-check before the container starts.
for filename in ["README.md", "Dockerfile", "app.py", "requirements.txt"]:
    local_path = os.path.join(DEPLOYMENT_DIR, filename)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=space_repo_id,
        repo_type="space",
        token=HF_TOKEN
    )
    print(f"✅ Uploaded {filename} → https://huggingface.co/spaces/{space_repo_id}")

print(f"\n✅ Step 6 (Hosting) complete.")
print(f"   Live app: https://huggingface.co/spaces/{space_repo_id}")
print(f"   (Space will rebuild in ~3-5 minutes if files changed)")
