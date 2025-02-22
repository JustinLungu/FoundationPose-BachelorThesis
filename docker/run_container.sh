#!/bin/bash

# Remove any existing container "foundationpose"
docker rm -f foundationpose

# Hardcode the project directory
PROJECT_DIR="/home/justin/thesis/FoundationPose"
echo "Project directory: $PROJECT_DIR"

# Enable GUI access for the container
xhost +local:docker

# Run the Docker container
docker run --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it \
  --network=host \
  --name foundationpose \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $PROJECT_DIR:/app \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  --ipc=host \
  -e DISPLAY=${DISPLAY} \
  foundationpose:latest \
  bash -c "cd /app && bash"
