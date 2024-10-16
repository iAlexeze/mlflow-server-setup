# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /mlflow_auth

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining contents into the container
COPY . .

# Set the default command to run the model training script
CMD ["python", "mlflow_auth.py"]

