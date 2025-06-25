#!/bin/bash

# # Define an array of image files with the same prompt for each object type
# declare -A images_and_prompts=(
#     #["load/images/internet/apple_1.png"]="a fresh apple"
#     #["load/images/internet/apple_2.png"]="a fresh apple"
#     #["load/images/internet/banana_1.png"]="a ripe banana"
#     #["load/images/internet/banana_2.png"]="a ripe banana"
#     #["load/images/internet/butter_knife_1.png"]="a butter knife"
#     #["load/images/internet/butter_knife_2.png"]="a butter knife"
#     #["load/images/internet/keyboard_1.png"]="a keyboard"
#     #["load/images/internet/keyboard_2.png"]="a keyboard"
#     #["load/images/stress_test/internet/banana.png"]="Banana (Curved Yellow Banana)."
#     #["load/images/internet/stapler_1.png"]="a stapler"
#     #["load/images/internet/stapler_2.png"]="a stapler"
#     #["load/images/internet/stapler_3.png"]="a stapler"
# )

# # Loop through the array and run the command for each image
# for image in "${!images_and_prompts[@]}"
# do
#     prompt=${images_and_prompts[$image]}
#     echo "Running Magic123 for $image with prompt: \"$prompt\""
#     python launch.py --config configs/magic123-coarse-sd.yaml --train --gpu 0 data.image_path=$image system.prompt_processor.prompt="$prompt"
# done



