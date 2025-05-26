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

# Default prompt
DEFAULT_PROMPT="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
PROMPT="${1:-$DEFAULT_PROMPT}"
# ===================================

echo "Cleaning up old container (if it exists)..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "Running container..."
docker run -it --rm \
  --gpus all \
  --name $CONTAINER_NAME \
  --network=host \
  -v $HOST_WORKSPACE:$CONTAINER_WORKSPACE \
  -v $HOST_CACHE_DIR:$CONTAINER_CACHE_DIR \
  -v $HOST_OUTPUTS:$CONTAINER_OUTPUTS \
  -e HF_HOME=/home/dreamer/.cache/huggingface \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  $IMAGE_NAME \
  bash -c "
      set -e
      cd /workspace

      echo '>>> Running training...'
      python launch.py \
        --config configs/dreamfusion-sd.yaml \
        --train --gpu 0 \
        system.prompt_processor.prompt=\"$PROMPT\"

      echo '>>> Finding latest trial directory from ThreeStudio outputs...'
      TRIAL_DIR=\$(ls -td outputs/dreamfusion-sd/* | head -n 1)
      echo \"Found trial directory: \$TRIAL_DIR\"

      echo '>>> Exporting model...'
      TORCH_LOAD_WEIGHTS_ONLY=0 python launch.py \
        --config \"\$TRIAL_DIR/configs/parsed.yaml\" \
        --export --gpu 0 \
        resume=\"\$TRIAL_DIR/ckpts/last.ckpt\" \
        system.exporter_type=mesh-exporter

      echo '>>> Export complete.'
    "