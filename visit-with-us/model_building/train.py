import pandas as pd
import xgboost as xgb
import joblib
import os

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ---------------------------
# Load dataset
# ---------------------------
dataset = load_dataset("ksricheenu/customer-tourism-prediction-dataset")
df = dataset["train"].to_pandas()

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# ---------------------------
# Identify column types
# ---------------------------
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

# ---------------------------
# Preprocessing
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols),
    ]
)

# ---------------------------
# Handle class imbalance
# ---------------------------
class_weight = y.value_counts()[0] / y.value_counts()[1]

# ---------------------------
# Model Pipeline
# ---------------------------
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "xgbclassifier",
            xgb.XGBClassifier(
                scale_pos_weight=class_weight,
                random_state=42,
                tree_method="hist",
                eval_metric="logloss",
            ),
        ),
    ]
)

# ---------------------------
# Grid Search
# ---------------------------
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1
)

grid.fit(X, y)

best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# ---------------------------
# Evaluation
# ---------------------------
print(classification_report(y, best_model.predict(X)))

# ---------------------------
# Save Model
# ---------------------------
model_file = "best_tourism_targeting_model_v1.joblib"
joblib.dump(best_model, model_file)

# ---------------------------
# Upload to HF
# ---------------------------
repo_id = "ksricheenu/customer-tourism-prediction-model"
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo=model_file,
    repo_id=repo_id,
    repo_type="model",
)

print("Model uploaded successfully")
