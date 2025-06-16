#!/bin/bash

# Define base folders containing models
BASE_FOLDERS=("outputs/magic123-coarse-sd")

# Number of GPUs available (modify based on SLURM request)
NUM_GPUS=2  # Set to the number of GPUs you requested in SLURM

# GPU assignment counter
GPU_INDEX=0

# Function to get the next available GPU
get_next_gpu() {
    GPU_INDEX=$(( (GPU_INDEX + 1) % NUM_GPUS ))
    echo $GPU_INDEX
}

# Export Loop: Use multiple GPUs for faster processing
for BASE_FOLDER in "${BASE_FOLDERS[@]}"; do
    echo "Processing folder: $BASE_FOLDER"

    # Find all subdirectories inside the base folder
    SUBFOLDERS=$(find "$BASE_FOLDER" -mindepth 1 -maxdepth 1 -type d)

    # Parallel Export with GPU distribution
    for FOLDER in $SUBFOLDERS; do
        CONFIG_PATH="$FOLDER/configs/parsed.yaml"
        CHECKPOINT_PATH="$FOLDER/ckpts/last.ckpt"

        if [[ -f "$CONFIG_PATH" && -f "$CHECKPOINT_PATH" ]]; then
            GPU_TO_USE=$(get_next_gpu)
            echo "Exporting model from: $FOLDER using checkpoint: $CHECKPOINT_PATH on GPU $GPU_TO_USE"

            # Run the export in parallel, distributing jobs across GPUs
            (
                python launch.py --config "$CONFIG_PATH" --export --gpu $GPU_TO_USE \
                    resume="$CHECKPOINT_PATH" \
                    system.exporter_type=mesh-exporter \
                    system.geometry.isosurface_method=mc-cpu \
                    system.exporter.fmt=obj-mtl \
                    system.geometry.isosurface_threshold=5. \
                    system.geometry.isosurface_resolution=256 \
                    system.exporter.save_uv=True \
                    system.exporter.save_texture=True \
                    system.exporter.texture_format=png \
                    system.exporter.texture_size=1024

                echo "Finished exporting: $FOLDER on GPU $GPU_TO_USE"
            ) &

            # Limit the number of parallel jobs to NUM_GPUS
            if [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; then
                wait -n
            fi
        else
            echo "Skipping $FOLDER (config or checkpoint file missing)."
        fi
    done

    wait  # Ensure all exports finish before moving to the next base folder
done

echo "All exports completed! Check your output directories for the OBJ, MTL, and texture files."
