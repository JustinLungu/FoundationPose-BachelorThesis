#!/bin/bash

# # Path to the folder containing the images
# #image_folder="./load/images/internet/zero"
# image_folder="./load/images/stress_test/internet"

# # Get all image file paths in the folder
# #images=($(ls $image_folder/*.png))
# images=($(ls $image_folder/banana.png))

# # Loop through the images and run the command
# for image in "${images[@]}"
# do
#     echo "Running Stable Zero123 for $image"
#     python launch.py --config configs/stable-zero123.yaml --train --gpu 0 data.image_path=$image
# done


BASE_FOLDERS=("outputs/zero123-sai")

# Loop through each base folder and process its subdirectories
for BASE_FOLDER in "${BASE_FOLDERS[@]}"; do
    echo "Processing folder: $BASE_FOLDER"

    # Find all subdirectories inside the base folder and store them in an array
    readarray -t SUBFOLDERS < <(find "$BASE_FOLDER" -mindepth 1 -maxdepth 1 -type d)

    # Run exports one by one (sequentially)
    for FOLDER in "${SUBFOLDERS[@]}"; do
        # Convert to absolute paths
        CONFIG_PATH="$(realpath "$FOLDER/configs/parsed.yaml")"
        CHECKPOINT_PATH="$(realpath "$FOLDER/ckpts/last.ckpt")"

        # Debugging: Print paths to verify correctness
        echo "Checking: $CONFIG_PATH and $CHECKPOINT_PATH"

        # Check if config and checkpoint files exist before running
        if [[ -f "$CONFIG_PATH" && -f "$CHECKPOINT_PATH" ]]; then
            echo "Exporting model from: $FOLDER using checkpoint: $CHECKPOINT_PATH"

            # Run the export command sequentially (no '&' at the end)
            python launch.py --config "$CONFIG_PATH" --export --gpu 0 \
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

            echo "Finished exporting: $FOLDER"
        else
            echo "Skipping $FOLDER (config or checkpoint file missing)."
        fi
    done
done

# Print a success message
echo "All exports completed! Check your output directories for the OBJ, MTL, and texture files."
