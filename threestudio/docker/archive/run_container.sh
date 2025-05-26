#!/bin/bash

PROMPT=${1:-"a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"}
echo ">>> Launching ThreeStudio with full GPU access and prompt: $PROMPT"

# Remove existing container if it's running
docker rm -f threestudio 2>/dev/null || true

# Start training and exporting inside the container
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
  bash -c "
    set -e
    cd /home/dreamer/threestudio

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
  "