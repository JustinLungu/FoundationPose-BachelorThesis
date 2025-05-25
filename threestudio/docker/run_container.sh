#!/bin/bash

# ========== CONFIGURATION ==========
IMAGE_NAME="threestudio_custom"
CONTAINER_NAME="threestudio"

# Path on the HOST where the project is located
HOST_WORKSPACE="/home/justin/thesis/FoundationPose-BachelorThesis/threestudio"
CONTAINER_WORKSPACE="/workspace"

# HuggingFace and other caches
HOST_CACHE_DIR="/home/justin/.cache/huggingface"
CONTAINER_CACHE_DIR="/home/dreamer/.cache/huggingface"
# ===================================

echo "Cleaning up old container (if it exists)..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

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
  -v $HOST_WORKSPACE:$HOME/threestudio \
  -v $HOST_CACHE_DIR:$CONTAINER_CACHE_DIR \
  -v /var/run/docker.sock:/var/run/docker.sock \
  $IMAGE_NAME \
  bash
