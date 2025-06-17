#!/bin/bash

# Path to the folder containing the images
image_folder="./load/images/linemod/cropped"

# Get all image file paths in the folder
images=($(ls $image_folder/*.png))

# Loop through the images and run the command
for image in "${images[@]}"
do
    echo "Running Stable Zero123 for $image"
    python launch.py --config configs/stable-zero123.yaml --train --gpu 0 data.image_path=$image
done