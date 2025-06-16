#!/bin/bash

# Ensure the script is executable: chmod +x generate_3d_models.sh

# Define an array of prompts
declare -a prompts=(
    #"Bored Gorilla. A stylized seated gorilla figurine with short, thick limbs, a rounded body, and a smooth head. The figure is red, has minimal facial features, and is leaning slightly forward."
    #"Rubber Duck. A small yellow rubber duck with a round body, a slightly raised tail, and molded wings. It has large circular eyes and an orange beak that is slightly open."
    #"Pink Kitten. A small pink cat figurine with a large rounded head, short legs, and a raised tail. The cat has small black eyes, a tiny nose, and is in a walking pose."
    #"Bosch Drill. A cordless power drill with a green body, black drill bit, and a T-shaped handle. The drill has a cylindrical chuck and a trigger button near the top."
    #"Camera. A DSLR camera with a black and silver body, a large zoom lens with a ridged focus ring. It has a right-hand grip, a shutter button on top, and a visible hot shoe mount."
    #"Egg Carton. A closed rectangular egg carton with a curved top and two visible egg-shaped protrusions. The carton has small locking tabs on the front."
    "Curved Yellow Banana."
)


# Iterate over each prompt and call the DeepFloyd IF script
for prompt in "${prompts[@]}"
do
    echo "Generating 3D model for: $prompt"
    python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 \
        system.prompt_processor.prompt="$prompt" system.background.random_aug=true
    echo "Completed generation for: $prompt"
done
