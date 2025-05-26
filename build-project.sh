#!/usr/bin/env bash

# ========== CONFIGURATION ==========
IMAGE_NAME="threestudio_custom"
CONTAINER_NAME="threestudio"
DOCKERFILE_DIR="threestudio/docker"


echo "Building Docker image for threestudio..."
docker build -t $IMAGE_NAME $DOCKERFILE_DIR


echo
echo "Building Docker image for FoundationPose..."
docker build \
  -t foundationpose-with-docker:latest \
  -f FoundationPose/docker/Dockerfile \
  FoundationPose

echo
echo "Build complete! You can now run ./run-project.sh"
