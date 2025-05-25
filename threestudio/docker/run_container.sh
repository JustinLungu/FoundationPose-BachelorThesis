#!/bin/bash

# ========== CONFIGURATION ==========
IMAGE_NAME="threestudio_custom"
CONTAINER_NAME="threestudio"

# Host paths
HOST_USER_HOME="/home/justin"
HOST_PROJECT_ROOT="$HOST_USER_HOME/thesis/FoundationPose-BachelorThesis"
HOST_WORKSPACE="$HOST_PROJECT_ROOT/threestudio"
HOST_CACHE_DIR="$HOST_USER_HOME/.cache/huggingface"

# Container paths
CONTAINER_WORKSPACE="/workspace"
CONTAINER_CACHE_DIR="/home/dreamer/.cache/huggingface"
# ===================================

echo "Cleaning up old container (if it exists)..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "Running container..."
docker run -it --rm \
  --gpus all \
  --name $CONTAINER_NAME \
  --network=host \
  -e PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:64 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $HOST_WORKSPACE:$CONTAINER_WORKSPACE \
  -v $HOST_CACHE_DIR:$CONTAINER_CACHE_DIR \
  $IMAGE_NAME \
  bash
