import pandas as pd
import xgboost as xgb
import joblib
import os

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ---------------------------
# Load datasets from HF
# ---------------------------
Xtrain_path = "hf://datasets/ksricheenu/customer-tourism-prediction-dataset/Xtrain.csv"
Xtest_path  = "hf://datasets/ksricheenu/customer-tourism-prediction-dataset/Xtest.csv"
ytrain_path = "hf://datasets/ksricheenu/customer-tourism-prediction-dataset/ytrain.csv"
ytest_path  = "hf://datasets/ksricheenu/customer-tourism-prediction-dataset/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()

# ---------------------------
# Handle class imbalance
# ---------------------------
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ---------------------------
# Model + GridSearch
# ---------------------------
model = make_pipeline(
    xgb.XGBClassifier(
        scale_pos_weight=class_weight,
        random_state=42,
        tree_method="hist"
    )
)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

grid = GridSearchCV(
    model,
    param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1
)

grid.fit(Xtrain, ytrain)

best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# ---------------------------
# Evaluation
# ---------------------------
print("\nTrain Report")
print(classification_report(ytrain, best_model.predict(Xtrain)))

print("\nTest Report")
print(classification_report(ytest, best_model.predict(Xtest)))

# ---------------------------
# Save model
# ---------------------------
model_file = "best_tourism_targeting_model_v1.joblib"
joblib.dump(best_model, model_file)

# ---------------------------
# Upload model to HF
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
    repo_type="model"
)

print("✅ Model uploaded successfully")
