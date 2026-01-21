# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb   #Trying this usecase using XGboost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

# for model serialization
import joblib

# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/ksricheenu/customer-tourism-prediction-model/Xtrain.csv"
Xtest_path = "hf://datasets/ksricheenucustomer-tourism-prediction-model/Xtest.csv"
ytrain_path = "hf://datasets/ksricheenu/customer-tourism-prediction-model/ytrain.csv"
ytest_path = "hf://datasets/ksricheenu/customer-tourism-prediction-model/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Class weight to handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

#As the categorical columns are already label encoded and other numeric columns are also not categorical we don't need to do any more one hot encoding.

# Creating pipeline
model_pipeline = make_pipeline(
    xgb.XGBClassifier(
        scale_pos_weight=class_weight,
        random_state=42,
        tree_method="hist"
    )
)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "best_tourism_targeting_model_v1.joblib")

# Upload to Hugging Face
repo_id = "ksricheenu/customer-tourism-prediction-model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_tourism_targeting_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_tourism_targeting_model_v1.joblib",
    path_in_repo="best_tourism_targeting_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
