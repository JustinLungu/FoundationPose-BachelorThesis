#!/bin/bash

# Step 1: Pull foundationpose image (only if not already pulled)
echo "Pulling FoundationPose image..."
docker pull wenbowen123/foundationpose
docker tag wenbowen123/foundationpose foundationpose

# Step 2: Build threestudio image from your Dockerfile
echo "Building threestudio container..."
docker build -t threestudio ./threestudio/docker

# Step 3: Run foundationpose container using your run_container.sh
echo "Launching FoundationPose container..."
bash FoundationPose/docker/run_container.sh
