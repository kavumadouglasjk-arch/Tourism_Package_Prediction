# train.py
# Loads train/test data from HF Hub, trains an XGBoost classifier with
# GridSearchCV hyperparameter tuning, logs all params and metrics to MLflow,
# and registers the best model to the HF Model Hub.

from huggingface_hub import HfApi, create_repo, hf_hub_download
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
HF_TOKEN         = os.environ["HF_TOKEN"]
HF_USER          = "kavumadouglas"
DATA_REPO_ID     = f"{HF_USER}/tourism"
MODEL_REPO_ID    = f"{HF_USER}/tourism-prediction-model"
TARGET           = "ProdTaken"
LOCAL_MODEL_DIR  = "tourism_project/model_building"
LOCAL_MODEL_PATH = f"{LOCAL_MODEL_DIR}/best_model.joblib"

# Trusted types required by MLflow skops serialisation for XGBoost pipelines.
# Without this list, mlflow.sklearn.log_model raises UntrustedTypesFoundException.
SKOPS_TRUSTED_TYPES = [
    "sklearn.compose._column_transformer._RemainderColsList",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tourism_package_prediction")

# ── 1. Load train and test data from HF dataset repo ─────────
print("⏳ Downloading train/test data from Hugging Face Hub...")
train_path = hf_hub_download(
    repo_id=DATA_REPO_ID, filename="train.csv",
    repo_type="dataset", token=HF_TOKEN
)
test_path = hf_hub_download(
    repo_id=DATA_REPO_ID, filename="test.csv",
    repo_type="dataset", token=HF_TOKEN
)

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
print(f"✅ Loaded — train: {train_df.shape[0]} rows | test: {test_df.shape[0]} rows")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

# ── 2. Preprocessing pipeline ────────────────────────────────
# One-hot encode all categorical columns; numeric columns pass through unchanged
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# ── 3. Define model and hyperparameter grid ───────────────────
# Algorithm: XGBoost (approved per rubric alongside Decision Tree,
# Random Forest, AdaBoost, Gradient Boosting, and Bagging)
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   xgb)
])

param_grid = {
    "classifier__n_estimators":  [100, 200],
    "classifier__max_depth":     [3, 5, 7],
    "classifier__learning_rate": [0.05, 0.1],
}

# ── 4. Tune model and log all parameters and metrics to MLflow
print("⏳ Running GridSearchCV — 12 candidates × 5 folds = 60 fits...")
with mlflow.start_run(run_name="xgboost_gridsearch"):

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",    # f1 chosen due to class imbalance (19% positive class)
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log all tuned hyperparameters to MLflow
    mlflow.log_params(best_params)
    print(f"\n✅ Best hyperparameters logged to MLflow:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # ── 5. Evaluate model performance ────────────────────────
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    # Log all evaluation metrics to MLflow
    mlflow.log_metrics(metrics)

    print("\n✅ Test set performance logged to MLflow:")
    for k, v in metrics.items():
        print(f"   {k:10s}: {v:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"   True Negatives : {cm[0,0]}  |  False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}  |  True Positives : {cm[1,1]}")

    # Log the trained pipeline as an MLflow model artifact.
    # skops_trusted_types whitelists XGBoost and sklearn internals to avoid
    # UntrustedTypesFoundException raised by MLflow skops serialisation.
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        skops_trusted_types=SKOPS_TRUSTED_TYPES
    )
    run_id = mlflow.active_run().info.run_id
    print(f"\n✅ MLflow model artifact logged (run_id: {run_id})")

# ── 6. Save the best model locally with joblib ───────────────
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
joblib.dump(best_model, LOCAL_MODEL_PATH)
print(f"\n✅ Best model saved locally: {LOCAL_MODEL_PATH}")

# ── 7. Register the model in the Hugging Face Model Hub ──────
api = HfApi()

create_repo(
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    exist_ok=True
)

api.upload_file(
    path_or_fileobj=LOCAL_MODEL_PATH,
    path_in_repo="best_model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)
print(f"✅ Model registered: https://huggingface.co/{MODEL_REPO_ID}")
print("\n✅ Step 4 (Model Training and Registration) complete.")
