  # Pull the official FoundationPose image
  FROM shingarey/foundationpose_custom_cuda121:latest

  # Set timezone and shell early
  ENV TZ=US/Pacific
  ENV SHELL=/bin/bash

  # Install Docker CLI & docker-compose
  USER root
  RUN apt-get update && \
      apt-get install -y --no-install-recommends \
          docker.io \
          curl \
          docker-compose && \
      rm -rf /var/lib/apt/lists/*

  # Install latest Compose v2
  RUN curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose && \
      chmod +x /usr/local/bin/docker-compose

  # Match docker group to host's GID (1001), create user, and assign permissions
  RUN groupadd -g 1001 docker || true && \
      groupadd -g 1000 foundation && \
      useradd -m -u 1000 -g foundation -s /bin/bash foundationpose && \
      usermod -aG docker foundationpose

  # Set bash as default shell for everything
  RUN ln -sf /bin/bash /bin/sh

  # Set working directory and add launcher script
  WORKDIR /workspace
  COPY docker/run_container.sh /usr/local/bin/run_container.sh
  RUN chmod +x /usr/local/bin/run_container.sh

  # Switch to non-root user
  #USER foundationpose
