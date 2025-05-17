Running the DooD

Prerequisites
This guide assumes the user is running Ubuntu with:

NVIDIA GPU + drivers installed

Docker installed with NVIDIA Container Toolkit

User added to the docker group (sudo usermod -aG docker $USER && newgrp docker)

./build-project.sh
./run_project.sh
cd ..
cd threestudio/docker
./run_container.sh

python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"



Note: onlu use the docker container when strcitly requiring pose estiamtion running. The evalution and preprocessing I would suggest to do it outside of the docker container and using a conda/venv.