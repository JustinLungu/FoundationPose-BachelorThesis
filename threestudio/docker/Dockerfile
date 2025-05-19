# Use CUDA 11.8 base image to match torch/cu118
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set user arguments
ARG USER_NAME=dreamer
ARG GROUP_NAME=dreamers
ARG UID=1001
ARG GID=1001

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/${USER_NAME}
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV TCNN_CUDA_ARCHITECTURES=89

# Fix HuggingFace + Matplotlib + Torch extension permission issues
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TORCH_HOME=/tmp/torch_extensions
ENV XDG_CACHE_HOME=/tmp

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates sudo wget ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    libeigen3-dev build-essential cmake libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python and pip defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    python -m pip install --upgrade pip

# Install PyTorch, torchvision, diffusers, huggingface_hub, accelerate to match host setup
RUN pip install \
    torch==2.5.1+cu118 \
    torchvision==0.20.1+cu118 \
    diffusers==0.19.3 \
    huggingface_hub==0.21.0 \
    accelerate==1.2.1 \
    torchmetrics==1.6.1 \
    xformers==0.0.29.post1 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Copy and install remaining dependencies (fallback-safe)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt || true

# Create user and make /tmp folders writable
RUN groupadd -g ${GID} ${GROUP_NAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /tmp/huggingface /tmp/matplotlib /tmp/torch_extensions && \
    chown -R ${UID}:${GID} /tmp/huggingface /tmp/matplotlib /tmp/torch_extensions

# Switch to non-root user
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/threestudio

# Default shell
CMD ["/bin/bash"]
