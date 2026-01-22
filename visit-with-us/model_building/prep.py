import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "visit-with-us/data/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier - customer id
df.drop(columns=['CustomerID'], inplace=True)

# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()
cols = ['TypeofContact', 'Occupation','Gender', 'ProductPitched', 'MaritalStatus', 'Designation' ]

for col in cols:
    df[col] = label_encoder.fit_transform(df[col])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

print("Train/test CSVs saved locally.")

# Upload files to Hugging Face (if HF_TOKEN is valid)
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],  # just the filename
            repo_id="ksricheenu/customer-tourism-prediction-model",
            repo_type="dataset",
        )
        print(f"Uploaded {file_path} to Hugging Face dataset repo.")
