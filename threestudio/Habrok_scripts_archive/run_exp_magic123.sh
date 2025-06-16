#!/bin/bash

# Define an array of image files with the same prompt for each object type
declare -A images_and_prompts=(
    ["load/images/linemod/gorilla/0000.png"]="Monkey. A stylized seated gorilla figurine with short, thick limbs, a rounded body, and a smooth head. The figure is red, has minimal facial features, and is leaning slightly forward."
    ["load/images/linemod/gorilla/0203.png"]="Monkey. A stylized seated gorilla figurine with short, thick limbs, a rounded body, and a smooth head. The figure is red, has minimal facial features, and is leaning slightly forward."
    ["load/images/linemod/gorilla/0297.png"]="Monkey. A stylized seated gorilla figurine with short, thick limbs, a rounded body, and a smooth head. The figure is red, has minimal facial features, and is leaning slightly forward."
    ["load/images/linemod/gorilla/1222.png"]="Monkey. A stylized seated gorilla figurine with short, thick limbs, a rounded body, and a smooth head. The figure is red, has minimal facial features, and is leaning slightly forward."

    )

# Loop through the array and run the command for each image
for image in "${!images_and_prompts[@]}"
do
    prompt=${images_and_prompts[$image]}
    echo "Running Magic123 for $image with prompt: \"$prompt\""
    python launch.py --config configs/magic123-coarse-sd.yaml --train --gpu 0 data.image_path=$image system.prompt_processor.prompt="$prompt"
done
