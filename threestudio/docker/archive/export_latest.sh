#!/bin/bash

echo ">>> Launching ThreeStudio export only..."

# Remove existing container if it's running
docker rm -f threestudio 2>/dev/null || true

# Run ThreeStudio container for export only
docker run --rm -it \
  --gpus all \
  --name threestudio \
  --network=host \
  -v /home/justin/thesis/FoundationPose-BachelorThesis/threestudio:/home/dreamer/threestudio \
  -v /home/justin/.cache/huggingface:/home/dreamer/.cache/huggingface \
  -e HF_HOME=/home/dreamer/.cache/huggingface \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  threestudio:latest \
  bash
