from mlflow import MlflowClient
from mlflow.server import get_app_client
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

# Set the MLflow tracking URI and model name from the environment variable
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.eclathealthcare.com/")

# Username and password for basic authentication
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "asalex") 
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "asalex")

response = requests.get(
    tracking_uri,
    auth=(MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD),
)

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
auth_client.create_user(username="user1", password="pw1")
auth_client.create_user(username="user2", password="pw2")

client = MlflowClient(tracking_uri=tracking_uri)
experiment_id = client.create_experiment(name="experiment")

auth_client.create_experiment_permission(
    experiment_id=experiment_id, username="user1", permission="READ"
)
