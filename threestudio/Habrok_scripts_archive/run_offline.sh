#!/bin/bash
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=magic123sd_%j.log

# modules loading and environment activation
module purge
module load CUDA/11.8.0
module load Python/3.10.8
module load GCC/11.3.0

export CUDA_HOME=/cvmfs/hpc.rug.nl/versions/2023.01/rocky8/x86_64/amd/zen3/software/CUDA/11.8.0 
export PATH=$CUDA_HOME/bin:$PATH 
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 
export CC=gcc export CXX=g++

nvcc --version
python --version
gcc --version
echo $CUDA_HOME
module list

source /home2/s4787641/threestudio/threestudio/bin/activate


# #========================= DREAMFUSION-FI ===================================
# Define an array of prompts
declare -a prompts=(
    #"Red Gorilla (Gorilla Figurine). A seated gorilla figurine with a round, bulky body, thick arms, and short legs. The head is smooth with minimal facial features."
    #"Yellow Duck (Rubber Duck). A small rubber duck with a rounded body, a slightly raised tail, and molded wings. The head is small and smooth, with a simple beak."
    #"Pink Cat (Toy Cat Figurine). A small cat figurine with a rounded head, short legs, and a slightly curved tail. The body is smooth and compact."
    #"Drill (Cordless Power Drill). A T-shaped power drill with a short cylindrical front for the drill bit and a boxy handle. The structure is compact and symmetrical."
    #"Camera (DSLR Camera). A rectangular DSLR camera with cylindrical lens at the front. The top has a small bump for controls, and the overall structure is blocky and solid."
    #"Egg Carton (Closed Carton). A closed rectangular 6-egg carton. The structure is flat and symmetrical."

    "Banana (Curved Yellow Banana)." # A slightly curved yellow banana with a smooth peel and a tapered tip at both ends. One end is slightly darker (stem), and the surface may have a few natural spots. The overall shape is organic and asymmetric in 3D space."
    #"Mouse (Wireless Computer Mouse). A compact wireless mouse with a slightly domed top and a sleek, ergonomic shape. It has a central scroll wheel and two click buttons molded into a seamless top shell. The base is slightly flattened with a gentle taper."
    #"Scissors (Office Scissors). A pair of scissors with orange plastic handles and sharp metallic blades. The blades cross at a pivot point and taper to a sharp tip. The handles are symmetrical loops with a smooth ergonomic contour."
    #"Soda Can (Orange Soda Can). A standard aluminum soda can with a cylindrical shape, slightly indented top and bottom, and a pull tab. The surface features bright orange branding, with minimal bumps or texture variation. The form is vertically symmetrical."
    #"Stapler (Office Stapler). A black office stapler with a flat rectangular base and a top arm that lifts and presses down. The shape is angular and industrial, with visible segmentation between parts. The structure is blocky and asymmetrical in elevation."
)


# # Iterate over each prompt and call the DeepFloyd IF script
# for prompt in "${prompts[@]}"
# do
#     echo "Generating 3D model for: $prompt"
#     python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 \
#         system.prompt_processor.prompt="$prompt" system.background.random_aug=true
#     echo "Completed generation for: $prompt"
# done


# #launch the model
# #========================= ZERO123 ============================
# # Path to the folder containing the images
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



# #=========================== MAGIC123 ===================================
# Define an array of image files with the same prompt for each object type
# declare -A images_and_prompts=(
#     #["load/images/linemod/gpt/uneven/gorilla_GPT.png"]="Red Gorilla (Gorilla Figurine). A seated gorilla figurine with a round, bulky body, thick arms, and short legs. The head is smooth with minimal facial features."
#     #["load/images/linemod/gpt/uneven/duck_GPT.png"]="Yellow Duck (Rubber Duck). A small rubber duck with a rounded body, a slightly raised tail, and molded wings. The head is small and smooth, with a simple beak."
#     #["load/images/linemod/gpt/uneven/cat_GPT.png"]="Pink Cat (Toy Cat Figurine). A small cat figurine with a rounded head, short legs, and a slightly curved tail. The body is smooth and compact."
#     #["load/images/linemod/gpt/uneven/drill_GPT.png"]="Drill (Cordless Power Drill). A T-shaped power drill with a short cylindrical front for the drill bit and a boxy handle. The structure is compact and symmetrical."
#     #["load/images/linemod/gpt/uneven/camera_GPT.png"]="Camera (DSLR Camera). A rectangular DSLR camera with cylindrical lens at the front. The top has a small bump for controls, and the overall structure is blocky and solid."
#     #["load/images/linemod/gpt/uneven/eggs_GPT.png"]="Egg Carton (Closed Carton). A closed rectangular 6-egg carton.The structure is flat and symmetrical."
#     ["load/images/stress_test/internet/banana.png"]="Banana (Curved Yellow Banana)." #A slightly curved yellow banana with a smooth peel and a tapered tip at both ends. One end is slightly darker (stem), and the surface may have a few natural spots. The overall shape is organic and asymmetric in 3D space."
#     )

# # Loop through the array and run the command for each image
# for image in "${!images_and_prompts[@]}"
# do
#     prompt=${images_and_prompts[$image]}
#     echo "Running Magic123 for $image with prompt: \"$prompt\""
#     python launch.py --config configs/magic123-coarse-sd.yaml --train --gpu 0 data.image_path=$image system.prompt_processor.prompt="$prompt"
# done






# Define a list of base folders (modify this to include multiple sources)
# BASE_FOLDERS=("outputs/zero123-sai")
#BASE_FOLDERS=("outputs/magic123-coarse-sd")
BASE_FOLDERS=("outputs/dreamfusion-if")

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