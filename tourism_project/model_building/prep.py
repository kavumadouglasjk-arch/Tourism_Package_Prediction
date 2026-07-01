# prep.py
# Cleans the tourism dataset, splits into train/test sets,
# and uploads both back to the Hugging Face dataset repo.

from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────
HF_TOKEN       = os.environ["HF_TOKEN"]
HF_USER        = "kavumadouglas"
repo_id        = f"{HF_USER}/tourism"
LOCAL_DATA_DIR = "tourism_project/data"
TARGET         = "ProdTaken"

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# ── 1. Load tourism.csv from the HF dataset repo ─────────────
print("⏳ Downloading tourism.csv from Hugging Face Hub...")
csv_path = hf_hub_download(
    repo_id=repo_id, filename="tourism.csv",
    repo_type="dataset", token=HF_TOKEN
)
df = pd.read_csv(csv_path)
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Data Cleaning ─────────────────────────────────────────

# Drop columns with no predictive value
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"✅ Dropped columns: {drop_cols}")

# Fix Gender typo: "Fe Male" → "Female"
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
print("✅ Fixed Gender typo: 'Fe Male' → 'Female'")

# Merge duplicate MaritalStatus categories: "Unmarried" → "Single"
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
print("✅ Merged MaritalStatus: 'Unmarried' → 'Single'")

# Drop fully duplicate rows
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"✅ Removed {before - df.shape[0]} duplicate rows")

# Handle missing values — median for numeric, mode for categorical
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    if col != TARGET and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=["object"]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
print("✅ Missing values handled (median/mode imputation)")

print(f"\n📊 Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Target distribution:\n{df[TARGET].value_counts()}")

# ── 3. Train / Test Split (80/20, stratified on target) ──────
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.copy(); train_df[TARGET] = y_train
test_df  = X_test.copy();  test_df[TARGET]  = y_test
print(f"\n✅ Split: train={train_df.shape[0]} rows | test={test_df.shape[0]} rows")

# ── 4. Save splits locally ────────────────────────────────────
train_path = f"{LOCAL_DATA_DIR}/train.csv"
test_path  = f"{LOCAL_DATA_DIR}/test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"✅ Saved locally: train.csv and test.csv")

# ── 5. Upload splits back to the HF dataset repo ─────────────
api = HfApi()
for local_path, repo_filename in [(train_path, "train.csv"), (test_path, "test.csv")]:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"✅ Uploaded {repo_filename} → https://huggingface.co/datasets/{repo_id}")

print("\n✅ Step 3 (Data Preparation) complete.")
