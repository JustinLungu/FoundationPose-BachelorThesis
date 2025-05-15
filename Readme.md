Running the DooD

✅ Prerequisites
🛠 This guide assumes the user is running Ubuntu with:

NVIDIA GPU + drivers installed

Docker installed with NVIDIA Container Toolkit

User added to the docker group (sudo usermod -aG docker $USER && newgrp docker)

cd threestudio/docker
docker compose build
docker compose up -d
docker compose exec threestudio bash
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
docker compose stop
docker compose start

cd FoundationPose/docker/
docker build -t shingarey/foundationpose_custom_cuda121:latest .
bash run_container.sh
bash build_all.sh
docker ps
