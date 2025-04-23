#!/bin/bash

# Name for the image and container
IMAGE_NAME="thesis-env"
CONTAINER_NAME="thesis-container"

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Run the container with mounted code and interactive terminal
echo "Running Docker container: $CONTAINER_NAME..."
docker run --gpus all -it --rm \
  --name $CONTAINER_NAME \
  -v "$(pwd)":/app \
  --workdir /app \
  $IMAGE_NAME
