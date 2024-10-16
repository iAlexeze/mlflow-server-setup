import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
import os

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Set the MLflow tracking URI and model name from the environment variable
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment = os.getenv("EXPERIMENT_NAME")
model_name = os.getenv("MODEL_NAME")

# Username and password for basic authentication
mlflow_username = os.getenv("MLFLOW_USERNAME", "asalex")  # Default to 'asalex'
mlflow_password = os.getenv("MLFLOW_PASSWORD", "asalex")  # Default to 'asalex'

# Update the tracking URI with authentication
tracking_uri_with_auth = f"https://{mlflow_username}:{mlflow_password}@{tracking_uri}"

# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=2,
    n_redundant=8,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42
)

np.unique(y, return_counts=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)

# Configure MLflow
mlflow.set_tracking_uri(tracking_uri_with_auth)
mlflow.set_experiment(experiment)

with mlflow.start_run(run_name=model_name):
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(lr, "Logistic Regression")
    
