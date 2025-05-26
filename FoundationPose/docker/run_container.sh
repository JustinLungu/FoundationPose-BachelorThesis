#!/usr/bin/env bash
set -euo pipefail

# Remove any existing FoundationPose container
docker rm -f foundationpose 2>/dev/null || true

# Enable GUI access
xhost +local:docker

# Define absolute path to repo root
PROJECT_DIR="/home/$USER/thesis/FoundationPose-BachelorThesis"
echo "Project directory: $PROJECT_DIR"

# Run FoundationPose container
docker run --rm -it \
  --runtime=nvidia --gpus all \
  --group-add 1001 \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --network=host \
  --name foundationpose \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PROJECT_DIR":/app \
  -e DISPLAY=${DISPLAY} \
  -e CUDA_HOME=/usr/local/cuda \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e GIT_INDEX_FILE \
  foundationpose-with-docker:latest \
  bash -c "
    export HOST_USER_NAME=foundationpose;
    echo '>>> [FP] Dropping into FoundationPose shell...';
    cd /app/FoundationPose;
    exec bash"