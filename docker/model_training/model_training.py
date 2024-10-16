import numpy as np
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Load environment variables from .env file
load_dotenv()

# Set the MLflow tracking URI and model name from the environment variable
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
model_name = os.getenv("MODEL_NAME")
experiment = os.getenv("EXPERIMENT_NAME")

# Username and password for basic authentication
mlflow_username = os.getenv("MLFLOW_USERNAME", "asalex")  # Default to 'asalex'
mlflow_password = os.getenv("MLFLOW_PASSWORD", "asalex")  # Default to 'asalex'

# Update the tracking URI with authentication
tracking_uri_with_auth = f"https://{mlflow_username}:{mlflow_password}@{tracking_uri}"

# Configure MLflow
mlflow.set_tracking_uri(tracking_uri_with_auth)
mlflow.set_experiment(experiment)

if __name__ == "__main__":
    with mlflow.start_run(run_name=model_name):
        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        score = lr.score(X, y)
        print(f"Score: {score}")
        mlflow.log_metric("score", score)
        predictions = lr.predict(X)
        signature = infer_signature(X, predictions)
        mlflow.sklearn.log_model(lr, "model", signature=signature)
        print(f"Model saved in run {mlflow.active_run().info.run_uuid}")
