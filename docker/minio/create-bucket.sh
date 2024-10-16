#!/bin/bash

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."

# Loop until MinIO responds
while ! mc alias set myminio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}; do
    echo "MinIO is not ready yet. Waiting..."
    sleep 2  # Check every 2 seconds
done

#Configure MinIO Client
mc alias set minioserver http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Create the MLFlow bucket
mc mb minioserver/mlflow
