#!/bin/bash

# ========== CONFIGURATION ==========
IMAGE_NAME="threestudio_custom"
CONTAINER_NAME="threestudio"
CACHE_DIR="/home/justin/"

# Optional: directory where your Dockerfile is located
DOCKERFILE_DIR="."
# Optional: mount the project directory into the container
HOST_WORKSPACE="$(cd .. && pwd)"
CONTAINER_WORKSPACE="/workspace"
# ===================================

echo "Cleaning up old container (if it exists)..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# echo "Building Docker image..."
# docker build -t $IMAGE_NAME $DOCKERFILE_DIR

echo "Running container..."
docker run -it --rm \
  --gpus all \
  --name $CONTAINER_NAME \
  --privileged \
  --network=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v $HOST_WORKSPACE:$CONTAINER_WORKSPACE \
  -v $CACHE_DIR/.cache/huggingface:/home/dreamer/.cache/huggingface \
  -v /var/run/docker.sock:/var/run/docker.sock \
  $IMAGE_NAME \
  bash
